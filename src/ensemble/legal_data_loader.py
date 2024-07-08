import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import torch
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from utils import load_data
import argparse
from train_wrapper import SequenceClassificationModule
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
#from transformer_models import SequenceClassificationDataset


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
        )#.to(self.examples)
        return {"model_inputs": model_inputs, "raw_text": [i[0] for i in batch]}

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
        )#.to(self.examples)
        labels = torch.tensor([i[1] for i in batch])#.to(self.device)
        return {"model_inputs": model_inputs, "label": labels, "raw_text": [i[0] for i in batch]}


class TextDataModule(pl.LightningDataModule):
    def __init__(self, data_path, model_name, n_labels, batch_size=64, num_workers=1):
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
        if self.n_labels == 10000:
            str_label = "10k"
        elif self.n_labels == 20000:
            str_label = "20k"
        elif self.n_labels == 50000:
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
        # self.prepare_data()
        self.X_train = self.train.destination_context.tolist()
        self.y_train = [self.passage2labelid[row.passage_id] for _, row in self.train.iterrows()]

        self.X_val = self.val.destination_context.tolist()
        self.y_val = [self.passage2labelid[row.passage_id] for _, row in self.val.iterrows()]

        self.train_dataset = SequenceClassificationDataset(self.X_train, self.y_train, self.tokenizer)
        self.val_dataset = SequenceClassificationDataset(self.X_val, self.y_val, self.tokenizer)
        self.test_dataset = SequenceClassificationDatasetNoLabels(self.test.destination_context.tolist(), self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.train_dataset.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.val_dataset.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.test_dataset.collate_fn)
    
    #ToDo: Implement
    def predict_dataloader(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--only_prediction", default=None, type=str, help="???."
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="number of workers to load batches"
    )
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--n_labels", default=10000, type=str, help="")
    args = parser.parse_args()

    data_module = TextDataModule(
        data_path="data/",
        model_name=args.model_name,
        n_labels=args.n_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    ) 

    data_module.prepare_data()
    data_module.setup()
    args.len_train_dataloader = len(data_module.train_dataloader())
    # Example of using the datamodule to get a dataloader
    """
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print(batch["label"])
        break
    """
    api_key_wandb = "89dd0dde666ab90e0366c4fec54fe1a4f785f3ef"
    wandb.login(key=api_key_wandb)
    wandb_logger = WandbLogger(project='LePaRD_classification', entity='sup-res-dl', log_model='all')
    checkpoint_callback = ModelCheckpoint(every_n_epochs=args.num_epochs)
    #don't limit batches, breaks learning rate scheduler
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.pytorch.Trainer(limit_train_batches=9999999, limit_val_batches = 9999999, max_epochs=args.num_epochs,check_val_every_n_epoch=1,log_every_n_steps=20,logger=wandb_logger,callbacks=[checkpoint_callback,lr_monitor])
    trainer.fit(SequenceClassificationModule(args=args), data_module.train_dataloader(), data_module.val_dataloader())