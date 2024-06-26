import json
import random
import argparse
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    f1_score,
)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import os
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
# from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from sklearn import metrics


#Important: For now, avoid gradient acc
class SequenceClassificationModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model_name = args.model_name
        self.num_labels = args.n_labels
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon
        self.num_epochs = args.num_epochs
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.len_train_dataloader = args.len_train_dataloader

        self.save_hyperparameters()
    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self(**batch["model_inputs"], labels=batch["label"])
        loss = output.loss / self.gradient_accumulation_steps
        self.log('train_loss', loss)
        logits,targets = output.logits.detach().to('cpu').numpy(),batch["label"].to('cpu').numpy()
        top_1_accuracy = top_k_accuracy_score(targets, logits, k=1 , labels=np.arange(10000))
        top_5_accuracy = top_k_accuracy_score(targets, logits, k=5, labels=np.arange(10000))
        top_10_accuracy = top_k_accuracy_score(targets, logits, k=10, labels=np.arange(10000))
        self.log("top_1_train_accuracy", top_1_accuracy)
        self.log("top_5_train_accuracy", top_5_accuracy)
        self.log("top_10_train_accuracy", top_10_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            output = self.model(**batch["model_inputs"])
            logits = output.logits.detach().to('cpu').numpy()
            targets = batch["label"].to('cpu').numpy()
            #self.log("val_loss", output.loss)
            top_1_accuracy = top_k_accuracy_score(targets, logits, k=1 , labels=np.arange(10000))
            top_5_accuracy = top_k_accuracy_score(targets, logits, k=5, labels=np.arange(10000))
            top_10_accuracy = top_k_accuracy_score(targets, logits, k=10, labels=np.arange(10000))
            self.log("top_1_val_accuracy", top_1_accuracy)
            self.log("top_5_val_accuracy", top_5_accuracy)
            self.log("top_10_val_accuracy", top_10_accuracy)
    
    def predict(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            output = self.model(**batch["model_inputs"])
            return output

    #ToDo Implement testing
    """
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            self.model.eval()
    """
    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon
        )

        def lr_lambda(current_step: int):
            warmup_steps = 0
            total_steps = self.len_train_dataloader * self.num_epochs
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch' for epoch-level scheduler
                'frequency': 1,
                'reduce_on_plateau': False,
                'monitor': 'val_loss'
            }
        }
    
