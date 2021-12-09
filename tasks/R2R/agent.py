''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from transformers import T5Tokenizer
from env import R2RBatch
# from utils import padding_idx

tok = T5Tokenizer.from_pretrained("t5-small")
padding_idx = tok.pad_token_id

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        while True:
            for traj in self.teacher_rollout():
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break


class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        for t in range(30):
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, model, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.model = model
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        # randomly select one instrument among three
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]

        mask = (sorted_tensor == padding_idx)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def our_rollout(self, optimizer):
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
                                                                
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
            } for ob in perm_obs]

        # Our code for student forcing is here
        # for each action at time step i, we choose the t5 ith output as the predicted action
                        
        # monitering stop
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env
                                                
        # global action and image tensor
        # batch_actions = torch.zeros((batch_size, self.episode_len), dtype=torch.long, requires_grad=False).cuda()
        gt_actions = []
        #batch_img = torch.zeros((batch_size, self.episode_len, 2048), dtype=torch.float, requires_grad=False).cuda()
        batch_img = []

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).cuda()
        #batch_actions[:,0] = a_t

        # env action for interacting with environment
        self.loss = 0
        env_action = [None] * batch_size
        for i in range(self.episode_len):
            f_t = self._feature_variable(perm_obs) # Image features from obs
            batch_img.append(f_t)

            img_inputs = torch.stack(batch_img).reshape(batch_size, len(batch_img), -1)
            optimizer.zero_grad()
            logits = self.model(seq.cuda(), seq_mask.cuda(), img_inputs.cuda()) # [bs, action_len, 6]
            logits_t = logits[:,i] # predicted action in current time step, [bs, 6]

            '''
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logits_t[i, self.model_actions.index('forward')] = -float('inf')
            '''

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            #self.loss += self.criterion(logits_t, target)
            gt_actions.append(target)


            loss_t =  self.criterion(logits_t, target)
            self.loss += loss_t
            loss_t.backward()
            optimizer.step()

            # Determine next model inputs (student forcing)
            if self.feedback == 'teacher':
                a_t = target
            elif self.feedback == 'argmax':
                _,a_t = logits_t.max(1)
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logits_t, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample() 

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            # update env
            print ("env_action:", env_action)
            print ("target:", target)
            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break

        #img_inputs = torch.stack(batch_img).reshape(batch_size, len(batch_img), -1)
        #logits = self.model(seq.cuda(), seq_mask.cuda(), img_inputs.cuda())
        #self.loss = self.criterion(logits.reshape(batch_size*len(gt_actions), -1),
        #                        torch.flatten(torch.stack(gt_actions)))
        #self.losses.append(self.loss.item())

        self.losses.append(self.loss.item() / self.episode_len)
        return traj

    def teacher_rollout(self):
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Our code is here
        # input:
        # 1. tokenized instrument and masks for t5 encoder, [bs, text_max_len] = [100, 80]
        # 2. masked token
        # 3. ground true image features for t5 decoder, [bs, episode_len, img_feature] = [100, 20, 2048]
        # 4. ground true action for calculating loss including 'start', [bs, episode_len+1] = [100, 20+1]
        # output:
        # 1. batch_loss

        # monitering stop
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env
        # all action tensor
        batch_actions = torch.zeros((batch_size, self.episode_len), dtype=torch.long, 
                requires_grad=False).cuda()
        batch_img = torch.zeros((batch_size, self.episode_len, 2048), dtype=torch.float, 
                requires_grad=False).cuda()
        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
        requires_grad=False).cuda()
        #batch_actions[:,0] = a_t

        # env action for interacting with environment
        env_action = [None] * batch_size
        # get rest decoder image features and actions by guiding with gt path
        for i in range(self.episode_len):
            # get this time step gt action(target)
            target = self._teacher_action(perm_obs, ended)
            f_t = self._feature_variable(perm_obs) # Image features from obs

            # save action and img feature
            # if agent reach the end, save the last image feature
            batch_actions[:,i] = target
            batch_img[:,i] = f_t
            
            # update 'ended' list and make environment action
            a_t = target # teacher forcing
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        
        # train here
        logits = self.model(seq, seq_mask, batch_img)
        loss_x = logits.reshape(batch_size*self.episode_len, -1)
        loss_y = torch.flatten(batch_actions)
        self.loss = self.criterion(logits.reshape(batch_size*self.episode_len, -1), 
                torch.flatten(batch_actions))
        self.losses.append(self.loss.item())
        return traj


    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.model.train()
        else:
            self.model.eval()
        super(Seq2SeqAgent, self).test()

    def train(self, optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.model.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            optimizer.zero_grad()
            self.teacher_rollout()
            # self.our_rollout(optimizer)
            self.loss.backward()
            optimizer.step()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.model.state_dict(), encoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.model.load_state_dict(torch.load(encoder_path))
