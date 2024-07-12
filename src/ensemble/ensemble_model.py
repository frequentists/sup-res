import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from sklearn import metrics
import numpy as np
from ..colbert import ScoreColBERT

class EnsembleModel(pl.LightningModule):
    # note, that check_point path is
    def __init__(self, classifier_checkpoint, colbert_state):
        super(EnsembleModel, self).__init__()
        self.classifier = self.load_classifier(classifier_checkpoint)
        self.colbert_scorer = ScoreColBERT(colbert_state)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize trainable weights
        self.classifier_weight = nn.Parameter(torch.tensor(0.5))
        self.colbert_weight = nn.Parameter(torch.tensor(0.5))

    def load_classifier(self, checkpoint_path):
        # Load the classifier from the wandb checkpoint    
        api_key_wandb = "89dd0dde666ab90e0366c4fec54fe1a4f785f3ef"
        wandb.login(key=api_key_wandb)
        # I think we dont need id as an argument here...
        run = wandb.init(project="LePaRD_classification",)
        artifact = run.use_artifact(checkpoint_path, type='model')
        artifact_dir = artifact.download()
        classifier = torch.load(f'{artifact_dir}/model.ckpt')
        classifier.eval()
        return classifier

    def forward(self, x_raw_text, **x_tokenized):
        classifier_output = self.classifier(**x_tokenized)
        print(classifier_output)
        colbert_output = self.colbert_scorer.scores_colbert(x_raw_text, n=classifier_output.size(1))
        return self.classifier_weight * classifier_output + self.colbert_weight * colbert_output

    def training_step(self, batch, batch_idx):
        output = self(batch["raw_text"], **batch["model_inputs"])
        loss = self.criterion(output, batch["labels"])
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
        output = self(batch["raw_text"], **batch["model_inputs"])
        val_loss = self.criterion(output, batch["labels"])
        self.log('val_loss', val_loss)
        logits = output.logits.detach().to('cpu').numpy()
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
        test_loss = self.criterion(output, batch["labels"])
        self.log('test_loss', test_loss)
        logits = output.logits.detach().to('cpu').numpy()
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