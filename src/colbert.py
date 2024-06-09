import platform
import yaml, json
import torch
import numpy as np
import pandas as pd
from csv import DictReader
from ragatouille import RAGPretrainedModel, RAGTrainer
from sklearn.model_selection import StratifiedKFold


def load_yaml(yaml_file_path):
    """Load yaml file and return dictionary"""
    try:
        with open(yaml_file_path, "r") as stream:
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


class ColBERT:
    """train and index ColBERT on legal training data and use it for search"""

    def __init__(self) -> None:
        pass

    def load_data(self, path="./data/top_10k.csv.gz") -> None:
        df = pd.read_csv(path, compression="gzip")
        df = df.sample(frac=1.0, random_state=42)

        self.lab2id = {int(i): j for j, i in enumerate(df.passage_id.unique())}

        train_idx, dev_idx = int(len(df) * 0.9), int(len(df) * 0.95)

        self.train = df[:train_idx]
        self.dev = df[train_idx:dev_idx]
        self.test = df[dev_idx:]

    def load_passages(self, path="./data/passage_dict.json") -> None:
        """load dictionary of passage_ids and passages"""

        with open(path, "r") as f:
            self.passages = json.load(f)
        self.passages = self.passages["data"]
        self.passages = {key: self.passages[str(key)] for key in self.lab2id.keys()}

    def setup_model(self, train=False) -> None:
        if not train:
            self.model = RAGPretrainedModel.from_pretrained(("colbert-ir/colbertv2.0"))
            pass
        # trainer = RAGTrainer(
        #     model_name="Test_ColBERT",
        #     pretrained_model_name="colbert-ir/colbertv2.0",
        #     language_code="en",
        # )
        # trainer.prepare_training_data(
        #     raw_data=train,
        #     data_out_path="./data/",
        #     all_documents=documents,
        #     num_new_negatives=0,
        #     mine_hard_negatives=False,
        # )
        # trainer.train(
        #     batch_size=32,
        #     nbits=4,  # How many bits will the trained model use when compressing indexes
        #     maxsteps=50,  # Maximum steps hard stop
        #     use_ib_negatives=True,  # Use in-batch negative to calculate loss
        #     dim=128,  # How many dimensions per embedding. 128 is the default and works well.
        #     learning_rate=5e-6,  # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
        #     doc_maxlen=64,  # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
        #     use_relu=False,  # Disable ReLU -- doesn't improve performance
        #     warmup_steps="auto",  # Defaults to 10%
        # )

    def index(self) -> None:
        self.model.index(
            collection=list(self.passages.values()),
            document_ids=list(map(str, list(self.passages.keys()))),
            # document_metadatas=document_metadata,
            index_name="LePaRD_pretrained",
            max_document_length=64,
            split_documents=False,
        )

    def device(self) -> str:
        if platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                return "mps"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            return "cuda"
