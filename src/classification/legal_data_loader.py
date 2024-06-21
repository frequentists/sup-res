import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from utils import load_data
# from transformer_models import SequenceClassificationDataset


class SequenceClassificationDatasetNoLabels(Dataset):
    def __init__(self, x, tokenizer):
        self.examples = x
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch):
        model_inputs = self.tokenizer(
            [i for i in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )#.to(self.device)
        return {"model_inputs": model_inputs}

class SequenceClassificationDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        self.examples = list(zip(x, y))
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch):
        # print (batch)
        model_inputs = self.tokenizer(
            [i[0] for i in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )#.to(self.device)
        labels = torch.tensor([i[1] for i in batch])#.to(self.device)
        return {"model_inputs": model_inputs, "label": labels}


class TextDataModule(pl.LightningDataModule):
    def __init__(self, data_path, model_name, n_labels, batch_size=64, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.model_name = model_name
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
    def prepare_data(self):
        # Download tokenizer if not already present
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, truncation_side="left")
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation_side="left")
        
        # Load data and passage2labelid mapping
        str_label = ""
        if self.n_labels == "10000":
            str_label = "10k"
        elif self.n_labels == "20000":
            str_label = "20k"
        elif self.n_labels == "50000":
            str_label = "50k"
        
        self.train, self.val, self.test, self.passage2labelid = load_data(
            dataframe=f"./data/top_{str_label}.csv.gz"
        )

        if not os.path.exists(f"{self.n_labels}_labelmap.json"):
            with open(f"{self.n_labels}_labelmap.json", "w") as outfile:
                json.dump(self.passage2labelid, outfile)
        
        self.label2passageid = {i: j for i, j in self.passage2labelid.items()}

    def setup(self, stage=None):
        # Split the data and create datasets
        self.prepare_data()
        self.X_train = self.train.destination_context.tolist()
        self.y_train = [self.passage2labelid[row.passage_id] for _, row in self.train.iterrows()]

        self.X_val = self.val.destination_context.tolist()
        self.y_val = [self.passage2labelid[row.passage_id] for _, row in self.val.iterrows()]

        self.train_dataset = SequenceClassificationDataset(self.X_train, self.y_train, self.tokenizer)
        self.val_dataset = SequenceClassificationDataset(self.X_val, self.y_val, self.tokenizer)
        self.test_dataset = SequenceClassificationDatasetNoLabels(self.test.destination_context.tolist(), self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--n_labels", default="10000", type=str, help="")
    args = parser.parse_args()

    data_module = TextDataModule(
        data_path="./data/",
        model_name=args.model_name,
        n_labels=args.n_labels,
        batch_size=args.batch_size
    )

    #data_module.prepare_data()
    data_module.setup()
    
    # Example of using the datamodule to get a dataloader
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print(batch)
        break
