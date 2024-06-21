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
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

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

        self.label2passageid = 

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self(**batch["model_inputs"], labels=batch["label"])
        loss = output.loss / self.gradient_accumulation_steps
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        output = model(**batch["model_inputs"])
        logits = output.logits
        targets.extend(batch["label"].float().tolist())
        outputs.extend(logits.argmax(dim=1).tolist())
        probs.extend(logits.softmax(dim=1)[:, 1].tolist())
        top_1_recall = accuracy_score(y_dev, outputs))
        top_5_recall = metric.top_k_accuracy_score()

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
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(self.train_dataloader()) * self.num_epochs
        )
        return [optimizer], [scheduler]