from transformers import T5ForConditionalGeneration
from transformers import T5Model
import torch
from torchvision.models import ResNet

model = T5Model.from_pretrained('t5-small')
input_ids = torch.LongTensor([[45, 32, 456, 34, 8, 57]])
attention_mask = torch.LongTensor([[1, 1, 1, 1, 1, 1]])

img_embeddings = torch.rand([1, 10, 512])
output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_inputs_embeds=img_embeddings, output_hidden_states = True)
print(type(output), output.last_hidden_state.shape)

#   1 2 3 4 5
# 0 1 2 3 4

# 5
# 0 1 2 3 4