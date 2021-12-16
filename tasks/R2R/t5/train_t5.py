
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import T5Tokenizer
from utils import padding_idx, timeSince
from env import R2RBatch
from model import CustomT5Model
from agent import Seq2SeqAgent
from eval import Evaluation

torch.cuda.empty_cache()
TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80

features = IMAGENET_FEATURES
batch_size = 8
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'teacher'  # teacher or sample
learning_rate = 0.001
weight_decay = 0.0005
n_iters = 5000 if feedback_method == 'teacher' else 20000
model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)

tok = T5Tokenizer.from_pretrained("t5-small")
tok.padding_side = "right"
padding_idx = tok.pad_token_id

def train(train_env, model, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    model = model.cuda()
    agent = Seq2SeqAgent(train_env, "", model, max_episode_len)
    print('Training with %s feedback' % feedback_method)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    for idx in range(0, n_iters, log_every):

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(optimizer, interval, feedback=feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric,val in score_summary.items():
                data_log['%s %s' % (env_name,metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)

        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR, model_prefix)
        df.to_csv(df_path)

        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        agent.save(enc_path, dec_path)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''

    setup()
    # Create a batch training environment that will also preprocess text
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    # Creat validation and test environments
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    print ("Build T5 model ...")
    model = CustomT5Model(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(), 2048)
    train(train_env, model, n_iters, val_envs=val_envs)


if __name__ == "__main__":
    train_val()
