import os.path

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from math import sqrt
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import config as CFG
from train import build_loaders
from clip import CLIPModel


def get_image_embeddings(test_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    test_loader = build_loaders(test_df, tokenizer, mode="valid")

    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)
    return model, torch.cat(test_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames, n=4):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(int(sqrt(n)), int(sqrt(n)), figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()


if __name__ == "__main__":
    test_df = pd.read_csv(os.path.join(CFG.test_dataset_path, CFG.test_dataset_filename))
    model, image_embeddings = get_image_embeddings(test_df, "best.pt")
    find_matches(model, image_embeddings,
                 query="girls",
                 image_filenames=test_df['image'].values, n=4)