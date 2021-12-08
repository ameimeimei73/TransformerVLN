import torch
import torch.nn as nn
from transformers import T5Model

class CustomT5Model(nn.Module):
    def __init__(self, action_size, image_feature_size):
        super(CustomT5Model, self).__init__()

        self.num_labels = action_size
        self.base_model = T5Model.from_pretrained('t5-small')  # small for test, should change to base
        self.model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']

        # Change beam search to greedy search
        self.base_model.config.num_beams = 1
        self.decoder_input = nn.Linear(in_features=action_size + image_feature_size, out_features=512)
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
        probs = torch.nn.functional.softmax(logits, dim=2)
        predictions = torch.argmax(probs, dim=2)

        return predictions

    def test(self, env, instruction_ids, max_steps):
        env.reset()
        action = torch.LongTensor(self.model_actions.index('<start>'))
        img_feature = env.step(action)
        decoder_input = torch.cat((action, img_feature), 2)
        attention_mask = torch.ones_like(instruction_ids)

        cur_step = 0
        output_action_seq = [action]
        # stop until reaches max step limits or model outputs <end> action
        while cur_step < max_steps and action is not self.model_actions.index('<end>'):
            output = self.base_model(
                instruction_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                decoder_inputs_embeds=decoder_input
            )

            hidden_states = output['last_hidden_state']
            action_prob = self.dense(hidden_states).softmax(dim=2)
            action = action_prob.argmax(dim=2)
            img_feature = env.step(action)
            next_input = torch.cat((action, img_feature), dim=2)
            decoder_input = torch.cat((decoder_input, next_input), dim=1)

            output_action_seq.append(action)
            cur_step += 1

        return output_action_seq


def main():
    model = CustomT5Model(8, 1024)
    input_ids = torch.LongTensor([[45, 32, 456, 34, 8, 57]])
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 1, 1]])
    actions = torch.rand([1, 10, 8])
    image_features = torch.rand([1, 10, 1024])
    output = model(input_ids=input_ids, attn_mask=attention_mask, actions=actions, image_features=image_features)
    print(output, output.shape)


if __name__ == "__main__":
    main()