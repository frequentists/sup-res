import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformer_models import SequenceClassificationDataset, evaluate_epoch, train_model
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from utils import *
import os
import yaml

# def train_model(trainset, model_name):
# def evaluate_epoch(model, dataset):
# class SequenceClassificationDataset(Dataset), 	def __init__(self, x, y, tokenizer):


def load_yaml(yaml_file_path):
    '''Load yaml file and return dictionary'''
    with open(yaml_file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def get_splits(X, y):
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    X = np.array(X)
    y = np.array(y)
    skf.get_n_splits(X, y)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        yield X_train, y_train, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--batch_size", default=64, type=int,
                help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--only_prediction", default=None, type=str,
                help="Epsilon for Adam optimizer.")
    parser.add_argument('--do_save', action='store_true')
    parser.add_argument("--n_labels", default="10000", type=str, help="")
    args = parser.parse_args()


    #args.save_path = "no-trainer-" + args.model_name + "-" + args.n_labels

    # args = load_yaml("config.yaml") # args is now a dictionary loaded from yaml file

    train, dev, test, passage2labelid = load_data(dataframe = "data/top_" + args.n_labels + "_training_data_NAACL.csv.gz")
    if os.path.exists(args.n_labels + "_labelmap.json"):
        pass
    else:
        with open(args.n_labels + "_labelmap.json", "w") as outfile:
            json.dump(passage2labelid, outfile)
    
    counter = 0
    to_save = []
    out = []

    args.save_path = "naacl-finetuned-" + args.model_name + "-" + str(args.n_labels)
                
    label2passageid = {i:j for i,j in passage2labelid.items()}

    X_train = train.destination_context.tolist()
    y_train = [passage2labelid[row.passage_id] for _, row in train.iterrows()]

    X_dev = dev.destination_context.tolist()
    y_dev = [passage2labelid[row.passage_id] for _, row in dev.iterrows()]
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side="left")
    except:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation_side="left")

    trainset = SequenceClassificationDataset(X_train, y_train, tokenizer)
    devset = SequenceClassificationDataset(X_dev, y_dev, tokenizer)
    model = train_model(trainset, args.model_name, args)

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    targets, outputs, probs = evaluate_epoch(model, devset, args)
    print ("accuracy", accuracy_score(y_dev, outputs))


    # sbatch --time=60 --mem-per-cpu=10000 --gpus=1 --gres=gpumem:10G --wrap="python climatebert_baseline.py" 
    # sbatch --time=60 --mem-per-cpu=10000 --gpus=1 --gres=gpumem:10G --wrap="python climatebert_baseline.py --model_name climatebert/distilroberta-base-climate-f"
    # sbatch --time=60 --mem-per-cpu=10000 --gpus=1 --gres=gpumem:10G --wrap="python climatebert_baseline.py --model_name roberta-base"
    # sbatch --time=60 --mem-per-cpu=10000 --gpus=1 --gres=gpumem:10G --wrap="python climatebert_baseline.py --model_name ../refactored-environmental-claims/src/envclaim-climatebert"

    # sbatch --time=1440 --mem-per-cpu=24000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 50000"
    
    # sbatch --time=240 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 10000 --model_name nlpaueb/legal-bert-base-uncased"
    """
sbatch --time=240 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 10000"
sbatch --time=240 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 10000 --model_name nlpaueb/legal-bert-base-uncased"

sbatch --time=1440 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 20000"
sbatch --time=1440 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 50000"    

sbatch --time=2880 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 20000 --model_name nlpaueb/legal-bert-base-uncased"
sbatch --time=2880 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 50000 --model_name nlpaueb/legal-bert-base-uncased"    

sbatch --time=1440 --mem-per-cpu=12000 --gpus=1 --gres=gpumem:10G --wrap="python src-classification/training_without_trainer.py --n_labels 10000 --model_name roberta-large --batch_size 16"

    """
 
