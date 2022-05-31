import cv2
import os
from transformers import DistilBertTokenizer, PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
import albumentations as A
import config as CFG
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import preprocess_dataset


class CLIPDataset(Dataset):
    def __init__(self, image_filenames: np.ndarray, captions: np.ndarray, tokenizer: PreTrainedTokenizer,
                 transform: A.Compose, enable_debug=False):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(self.captions, padding=True, truncation=True, max_length=CFG.max_length)
        self.transform = transform
        self.enable_debug = enable_debug

    def __getitem__(self, idx: int) -> dict:
        item = {
            key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{os.getcwd()}/{CFG.image_path}/{self.image_filenames[idx]}")
        if self.enable_debug:
            item['raw_img'] = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


def get_transform():
    transform = A.Compose(
        [
            A.Resize(CFG.size, CFG.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
    return transform


if __name__ == "__main__":
    preprocess_dataset()
    df = pd.read_csv(os.getcwd() + "/" + CFG.captions_path + "/captions.csv")
    image_filenames = df['image'].values
    captions = df['caption'].values
    transform = get_transform()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    clip_dataset = CLIPDataset(image_filenames=image_filenames, captions=captions, tokenizer=tokenizer,
                               transform=transform, enable_debug=True)
    sample = clip_dataset[10]
    print("Size of the transformed image: {}".format(sample['image'].size()))
    plt.imshow(sample['raw_img'])
    plt.title(sample['caption'])
    plt.show()
