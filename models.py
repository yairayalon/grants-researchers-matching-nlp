import torch
import torch.nn.functional as F
from transformers import AutoModel

from constants import device


# Mean Pooling - Take attention mask into account for correct averaging
# used for allmpnet model outputs
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


class SpecterModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained('allenai/specter').to(device)
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, 0, :]  # cls token


class AllMpnetModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        sent_emb = mean_pooling(output, input_ids['attention_mask'])
        sent_emb = F.normalize(sent_emb, p=2, dim=1)
        # return output.last_hidden_state[:, 0, :]  # cls token
        return sent_emb
