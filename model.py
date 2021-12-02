import torch
import torch.nn as nn
from transformers import T5Model

class CustomT5Model(nn.Module):
    def __init__(self, input_action_size, output_action_size, image_feature_size):
        super(CustomT5Model, self).__init__()

        self.num_labels = output_action_size
        self.base_model = T5Model.from_pretrained('t5-small')  # small for test, should change to base

        # Change beam search to greedy search
        self.base_model.config.num_beams = 1
        self.decoder_input = nn.Linear(in_features=input_action_size + image_feature_size, out_features=512)
        self.dense = nn.Linear(in_features=512, out_features=self.num_labels)

    def forward(self, input_ids, attn_mask, actions, image_features):
        # Create decoder input embedding
        concat_input = torch.cat((actions, image_features), 2)
        decoder_emb = self.decoder_input(concat_input)

        output = self.base_model(
            input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            decoder_inputs_embeds=decoder_emb
        )

        hidden_states = output['last_hidden_state']
        logits = self.dense(hidden_states)

        return logits

def main():
    model = CustomT5Model(8, 6, 1024)
    input_ids = torch.LongTensor([[45, 32, 456, 34, 8, 57]])
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 1, 1]])
    actions = torch.rand([1, 10, 8])
    image_features = torch.rand([1, 10, 1024])
    output = model(input_ids=input_ids, attn_mask=attention_mask, actions=actions, image_features=image_features)
    print(type(output), output.shape)


if __name__ == "__main__":
    main()