from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import *
from transformer_models import (
    SequenceClassificationDataset,
    evaluate_epoch,
    train_model,
)
from sklearn.metrics import classification_report, accuracy_score
import os


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
        ).to(self.device)
        return {"model_inputs": model_inputs}


def load_passages(fn="unique_citations_origin_only_valid_passages.jsonl"):
    with open(fn) as f:
        data = [json.loads(i) for i in f]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="trained-distilbert-base-uncased-25000/"
    )
    parser.add_argument("--n_labels", type=str, default="25000")
    parser.add_argument("--filename", type=str, default="data/test.jsonl")
    parser.add_argument("--to_evaluate", type=str, default="all")
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and tokenizer

    model_names = [
        "naacl-finetuned-distilbert-base-uncased-10000",
        "naacl-finetuned-distilbert-base-uncased-20000",
        "naacl-finetuned-distilbert-base-uncased-50000",
    ]

    # model_names = ["naacl-finetuned-nlpaueb/legal-bert-base-uncased-10000", "naacl-finetuned-nlpaueb/legal-bert-base-uncased-20000", "naacl-finetuned-nlpaueb/legal-bert-base-uncased-50000"]
    all_labels = ["10000", "20000", "50000"]

    for model_name, n_labels in zip(model_names, all_labels):
        print(model_name)
        train, dev, test, passage2labelid = load_data(
            dataframe="data/top_" + n_labels + "_training_data_NAACL.csv.gz"
        )
        label2passageid = {i: j for j, i in passage2labelid.items()}
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )

        X_dev = dev.destination_context.tolist()
        y_dev = [passage2labelid[row.passage_id] for _, row in dev.iterrows()]

        devset = SequenceClassificationDataset(X_dev, y_dev, tokenizer)
        targets, outputs, probs = evaluate_epoch(model, devset, args)
        print("accuracy", accuracy_score(y_dev, outputs))

        predict_dataset = SequenceClassificationDatasetNoLabels(
            dev.destination_context.tolist(), tokenizer
        )

        outputs = []
        probs = []
        hits = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(
                DataLoader(
                    predict_dataset,
                    batch_size=32,
                    collate_fn=predict_dataset.collate_fn,
                )
            ):
                # print (batch["model_inputs"]["input_ids"])
                # sys.exit(0)
                output = model(**batch["model_inputs"])
                argsorted = (
                    output.logits.argsort(dim=-1, descending=True).detach().cpu()
                )  # ok, argsort these
                hits_batch = []
                # ok, am I doing something wrong here?
                for i, j in zip(argsorted, output.logits):
                    i = [label2passageid[k.item()] for k in i[:10]]
                    hits_batch.append(i[:10])
                hits.extend(hits_batch)
                # print ("argsorted",[i[0] for i in hits_batch])
                # print ("argmax", output.logits.argmax(dim=-1))
                # ok, this is good
                # if len(hits) > 128:
                # 	break
        # true = [passage2labelid[i] for i in dev.passage_id]
        true = dev.passage_id.tolist()
        # print (np.array(hits).shape)
        for recall_at in [1, 5, 10]:
            is_correct = []
            for i, t in enumerate(true[: len(hits)]):
                predictions = hits[i][:recall_at]
                # predictions = [data[pred_doc]["passage_id"] for pred_doc in predictions]
                if t in predictions:
                    is_correct.append(1)
                else:
                    is_correct.append(0)
            print(
                "n labels dev",
                n_labels,
                "-- recall @ ",
                recall_at,
                np.round(np.mean(is_correct) * 100, 2),
            )

        with open(
            os.path.join(model_name, "predictions_devset_" + n_labels + ".json"), "w"
        ) as outfile:

            json.dump({"true": true, "hits": hits}, outfile)
        predict_dataset = SequenceClassificationDatasetNoLabels(
            test.destination_context.tolist(), tokenizer
        )

        outputs = []
        probs = []
        hits = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(
                DataLoader(
                    predict_dataset,
                    batch_size=32,
                    collate_fn=predict_dataset.collate_fn,
                )
            ):
                output = model(**batch["model_inputs"])
                argsorted = (
                    output.logits.argsort(dim=-1, descending=True).detach().cpu()
                )  # ok, argsort these
                hits_batch = []
                for i, j in zip(argsorted, output.logits):
                    i = [label2passageid[k.item()] for k in i[:10]]
                    hits_batch.append(i[:10])
                hits.extend(hits_batch)
        true = test.passage_id.tolist()
        # true = [passage2labelid[i] for i in test.passage_id]

        for recall_at in [1, 5, 10]:
            is_correct = []
            for i, t in enumerate(true):
                predictions = hits[i][:recall_at]
                # predictions = [data[pred_doc]["passage_id"] for pred_doc in predictions]
                if t in predictions:
                    is_correct.append(1)
                else:
                    is_correct.append(0)
            print(
                "n labels test",
                n_labels,
                "-- recall @ ",
                recall_at,
                np.round(np.mean(is_correct) * 100, 2),
            )
        with open(
            os.path.join(model_name, "predictions_testset_" + n_labels + ".json"), "w"
        ) as outfile:
            json.dump({"true": true, "hits": hits}, outfile)

# sbatch --time=120 --mem-per-cpu=10000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/inference_script_new.py"
