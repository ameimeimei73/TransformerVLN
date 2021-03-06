
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import T5Model
from transformers import BertModel

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class T5_Model(nn.Module):
    def __init__(self, input_action_size, output_action_size, image_feature_size):
        super(T5_Model, self).__init__()
        self.num_labels = input_action_size
        self.base_model = T5Model.from_pretrained('t5-small')  
        # Change beam search to greedy search
        self.base_model.config.num_beams = 1
        self.decoder_input = nn.Linear(in_features=image_feature_size, 
                out_features=512)
        self.dense = nn.Linear(in_features=512, out_features=output_action_size)

    def forward(self, input_ids, attn_mask, image_features):
        # Create decoder input embedding
        #concat_input = torch.cat((actions, image_features), 2)
        decoder_emb = self.decoder_input(image_features)

        output = self.base_model(
                input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                decoder_inputs_embeds=decoder_emb)

        hidden_states = output['last_hidden_state']
        logits = self.dense(hidden_states)
        probs = torch.nn.functional.softmax(logits, dim=2)

        return probs

class BERT_FC_Model(nn.Module):
    def __init__(self, input_action_size, output_action_size, image_feature_size):
        super(BERT_FC_Model, self).__init__()
        self.num_labels = input_action_size
        self.base_model = BertModel.from_pretrained("bert-base-uncased")

        # freeze parameter
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.action_fcs = nn.Sequential(
                        nn.Linear(768+2048, 128),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(128, output_action_size))

        #self.img_fc = nn.Linear(image_feature_size, 128)
        #self.pred_fc = nn.Linear(768, output_action_size)
        #self.relu = nn.ReLU()

        #torch.nn.init.xavier_uniform(self.img_fc.weight)
        #torch.nn.init.xavier_uniform(self.pred_fc.weight)

    def get_text_embedding(self, input_ids, attn_mask):
        logits = self.base_model(input_ids=input_ids.cuda(), attention_mask=attn_mask.cuda())
        cls_token = logits[0][:,0,:]
        return cls_token

    def forward(self, text_embed, image_features): 
        concat_input = torch.cat((text_embed, image_features), 1)
        logits = self.action_fcs(concat_input) # pos 0: [bs, 6]
        return logits


class BERT_LSTM_Model(nn.Module):
    '''
    BERT as encoder
    LSTM as decoder
    '''
    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size, dropout_ratio, feature_size=2048):
        super(BERT_LSTM_Model, self).__init__()
        # encoder part
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        # freeze model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # decoder part
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def encode(self, input_ids, attn_mask):
        logits = self.bert_model(input_ids=input_ids.cuda(), attention_mask=attn_mask.cuda())

        h_t = logits[0][:,0,:]
        c_t = logits[0][:,0,:]
        ctx = logits[0][:,1:-1,:]
        return ctx, h_t, c_t

    def decode(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()

        concat_input = torch.cat((action_embeds, feature), 1) # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        
        h_1,c_1 = self.lstm(drop, (h_0,c_0))
        h_1_drop = self.drop(h_1)

        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)   

        logit = self.decoder2action(h_tilde)
        return h_1,c_1,alpha,logit

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=2):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers, 
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''

        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
     
        mask = 1-mask # reverse mask in our model

        if mask is not None:
            # -Inf masking prior to the softmax 
            attn.data.masked_fill_(mask.bool(), -float('inf'))              
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size, 
                      dropout_ratio, feature_size=2048):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        concat_input = torch.cat((action_embeds, feature), 1) # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        h_1,c_1 = self.lstm(drop, (h_0,c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)        
        logit = self.decoder2action(h_tilde)
        return h_1,c_1,alpha,logit


