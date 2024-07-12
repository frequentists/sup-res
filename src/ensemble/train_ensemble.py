import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import torch
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from .utils import load_data
import argparse

from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
#from transformer_models import SequenceClassificationDataset
from ..colbert import DataState
from .legal_data_loader import SequenceClassificationDatasetNoLabels,SequenceClassificationDataset,TextDataModule
from .ensemble_model import EnsembleModel

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num_epochs", type=int, default=10)
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
    wandb_logger = WandbLogger(project='LePaRD_ensemble', entity='sup-res-dl', log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="top_1_val_accuracy",save_top_k = 1, mode = "max",auto_insert_metric_name=True, every_n_epochs=1,)
    trainer = pl.pytorch.Trainer(limit_train_batches=100, limit_val_batches = 100, max_epochs=args.num_epochs,check_val_every_n_epoch=1,log_every_n_steps=20,logger=wandb_logger,callbacks=[checkpoint_callback])
    
    state = DataState()
    checkpoint_path = "sup-resdl/LePaRD_classification/model-ymwdc5eu:v5" 
    trainer.fit(EnsembleModel(checkpoint_path, state), data_module.train_dataloader(), data_module.val_dataloader())