import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

import config as CFG
from dataset import CLIPDataset, get_transform
from clip import CLIPModel
from utils import AvgMeter, get_lr, preprocess_dataset
import matplotlib.pyplot as plt


def make_train_valid_test_dfs():
    preprocess_dataset()
    df = pd.read_csv(os.getcwd() + "/" + CFG.captions_path + "/captions.csv", sep=",")
    max_id = len(df) if not CFG.debug else CFG.debug_sample_size
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    test_ids = np.random.choice(image_ids, size=int(CFG.test_ratio * len(image_ids)), replace=False)
    _ids = [id_ for id_ in image_ids if id_ not in test_ids]
    train_ids = np.random.choice(_ids, size=int(CFG.train_valid_split * len(_ids)), replace=False)
    valid_ids = [id_ for id_ in _ids if id_ not in train_ids]
    test_df = df.iloc[test_ids].reset_index(drop=True)
    train_df = df.iloc[train_ids].reset_index(drop=True)
    valid_df = df.iloc[valid_ids].reset_index(drop=True)
    return train_df, valid_df, test_df


def build_loaders(df, tokenizer, mode="train"):
    image_filenames = df['image'].values
    captions = df['caption'].values
    transform = get_transform()
    dataset = CLIPDataset(image_filenames=image_filenames, captions=captions, tokenizer=tokenizer, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers,
                            shuffle=True if mode == "train" else False)
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df, valid_df, test_df = make_train_valid_test_dfs()

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    os.makedirs(CFG.test_dataset_path, exist_ok=True)
    test_df.to_csv(os.path.join(CFG.test_dataset_path, CFG.test_dataset_filename))
    os.makedirs(CFG.valid_dataset_path, exist_ok=True)
    valid_df.to_csv(os.path.join(CFG.valid_dataset_path, CFG.valid_dataset_filename))
    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"
    valid_loss_history = []
    train_loss_history = []
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        train_loss_history.append(train_loss.avg)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            valid_loss_history.append(valid_loss.avg)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
    epochs = range(CFG.epochs)
    plt.plot(epochs, train_loss_history, 'g', label='Training loss')
    plt.plot(epochs, valid_loss_history, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
