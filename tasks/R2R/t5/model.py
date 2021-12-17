
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from transformers import T5Model

# Modified T5 model with action as decoder input
class CustomT5Model(nn.Module):
    def __init__(self, input_action_size, output_action_size, image_feature_size):
        super(CustomT5Model, self).__init__()

        self.num_labels = output_action_size
        self.input_action_size = input_action_size
        self.base_model = T5Model.from_pretrained('t5-small')  # small for test, should change to base

        # Change beam search to greedy search
        self.base_model.config.num_beams = 1
        self.decoder_input = nn.Linear(in_features=self.input_action_size + image_feature_size, out_features=512)
        self.dense = nn.Linear(in_features=512, out_features=self.num_labels)
        # self.relu = nn.ReLU()

    def forward(self, input_ids, attn_mask, actions, image_features):
        # Create decoder input embedding
        concat_input = torch.cat((actions, image_features), 2)
        # decoder_emb = self.relu(self.decoder_input(concat_input))
        decoder_emb = self.decoder_input(concat_input)
        output = self.base_model(
            input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            decoder_inputs_embeds=decoder_emb)

        hidden_states = output['last_hidden_state']
        logits = self.dense(hidden_states)

        return logits

# Modified T5 model without action as decoder input
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
