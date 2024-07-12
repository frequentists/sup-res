import lightning as lt
import torch
import torch.nn as nn
import wandb
import os
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from sklearn import metrics
import numpy as np
from .train_wrapper import SequenceClassificationModule
from ..colbert import ScoreColBERT

class EnsembleModel(lt.LightningModule):
    # note, that check_point path is
    def __init__(self, classifier_checkpoint, colbert_state):
        super().__init__()
        self.classifier = self.load_classifier(classifier_checkpoint)
        self.colbert_scorer = ScoreColBERT(colbert_state)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize trainable weights
        self.classifier_weight = nn.Parameter(torch.tensor(0.5))
        self.colbert_weight = nn.Parameter(torch.tensor(0.5))

        self.save_hyperparameters()
    def load_classifier(self, checkpoint_path):
        # Load the classifier from the wandb checkpoint    
        api_key_wandb = "89dd0dde666ab90e0366c4fec54fe1a4f785f3ef"
        wandb.login(key=api_key_wandb)
        # I think we dont need id as an argument here...
        run = wandb.init(project="LePaRD_classification")
        artifact = run.use_artifact(checkpoint_path, type='model')
        artifact_dir = artifact.download()
        model_path = f"{artifact_dir}/model.ckpt"
        classifier = SequenceClassificationModule.load_from_checkpoint(model_path)
        classifier.model.eval()
        return classifier

    def forward(self, x_raw_text, **x_tokenized):
        classifier_output = self.classifier(**x_tokenized)
        colbert_output = torch.tensor(self.colbert_scorer.scores_colbert(x_raw_text, n=classifier_output.logits.size(1))).to(classifier_output.logits)
        return self.classifier_weight * classifier_output.logits + self.colbert_weight * colbert_output

    def training_step(self, batch, batch_idx):
        output = self(batch["raw_text"], **batch["model_inputs"])
        loss = self.criterion(output, batch["label"])
        self.log('train_loss', loss)
        logits,targets = output.detach().to('cpu').numpy(),batch["label"].to('cpu').numpy()
        top_1_accuracy = top_k_accuracy_score(targets, logits, k=1 , labels=np.arange(10000))
        top_5_accuracy = top_k_accuracy_score(targets, logits, k=5, labels=np.arange(10000))
        top_10_accuracy = top_k_accuracy_score(targets, logits, k=10, labels=np.arange(10000))
        self.log("top_1_train_accuracy", top_1_accuracy)
        self.log("top_5_train_accuracy", top_5_accuracy)
        self.log("top_10_train_accuracy", top_10_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch["raw_text"], **batch["model_inputs"])
        val_loss = self.criterion(output, batch["label"])
        self.log('val_loss', val_loss)
        logits = output.detach().to('cpu').numpy()
        targets = batch["label"].to('cpu').numpy()
        #self.log("val_loss", output.loss)
        top_1_accuracy = top_k_accuracy_score(targets, logits, k=1 , labels=np.arange(10000))
        top_5_accuracy = top_k_accuracy_score(targets, logits, k=5, labels=np.arange(10000))
        top_10_accuracy = top_k_accuracy_score(targets, logits, k=10, labels=np.arange(10000))
        self.log("top_1_val_accuracy", top_1_accuracy)
        self.log("top_5_val_accuracy", top_5_accuracy)
        self.log("top_10_val_accuracy", top_10_accuracy)
        return val_loss

    def test_step(self, batch, batch_idx):
        output = self(batch["raw_text"], **batch["model_inputs"])
        test_loss = self.criterion(output, batch["label"])
        self.log('test_loss', test_loss)
        logits = output.detach().to('cpu').numpy()
        targets = batch["label"].to('cpu').numpy()
        #self.log("test_loss", output.loss)
        top_1_accuracy = top_k_accuracy_score(targets, logits, k=1 , labels=np.arange(10000))
        top_5_accuracy = top_k_accuracy_score(targets, logits, k=5, labels=np.arange(10000))
        top_10_accuracy = top_k_accuracy_score(targets, logits, k=10, labels=np.arange(10000))
        self.log("top_1_test_accuracy", top_1_accuracy)
        self.log("top_5_test_accuracy", top_5_accuracy)
        self.log("top_10_test_accuracy", top_10_accuracy)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam([self.classifier_weight, self.colbert_weight], lr=1e-4)