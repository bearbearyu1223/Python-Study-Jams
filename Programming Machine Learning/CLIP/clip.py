import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, TextEncoder, ProjectionHead


def cross_entropy(preds, targets, reduction=None):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets*log_softmax(preds)).sum(1)
    if reduction is not None:
        return loss.mean()
    else:
        return loss


class CLIPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature, image_embedding=CFG.image_embedding,
                 text_embedding=CFG.text_embedding):
        super(CLIPModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(((images_similarity + texts_similarity)/2)/self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets)
        images_loss = cross_entropy(logits.T, targets.T)
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


if __name__ == "__main__":
    images = torch.rand(8, 3, 244, 244)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    model = CLIPModel()
    loss = model(batch)






