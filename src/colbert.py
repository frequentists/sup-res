import platform
import yaml, json
import torch
import numpy as np
import pandas as pd
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

    def load_data(
        self,
        path="./data/top_10k.csv.gz",
        keep_rows=["destination_context", "passage_id"],
    ) -> None:
        df = pd.read_csv(path, compression="gzip")
        df = df[keep_rows]
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        self.lab2id = {int(i): j for j, i in enumerate(df.passage_id.unique())}

        train_idx, dev_idx = int(len(df) * 0.9), int(len(df) * 0.95)
        # train_idx, dev_idx = 500, int(len(df) * 0.95)

        self.train = df[:train_idx]
        self.dev = df[train_idx:dev_idx]
        self.test = df[dev_idx:]

    def load_passages(self, path="./data/passage_dict.json") -> None:
        """load dictionary of passage_ids and passages"""

        with open(path, "r") as f:
            self.passages = json.load(f)
        self.passages = self.passages["data"]
        self.passages = {key: self.passages[str(key)] for key in self.lab2id.keys()}

    def setup_model(self, train=False, checkpoint=None) -> None:
        """setup model with colbert, either pretrained or finetuned"""

        if not train:
            self.model = RAGPretrainedModel.from_pretrained(
                pretrained_model_name_or_path=(
                    checkpoint if checkpoint else "colbert-ir/colbertv2.0"
                )
            )
            return
        self.model = RAGTrainer(
            model_name="fine-tuned", pretrained_model_name="colbert-ir/colbertv2.0"
        )
        self.model_finetune()

    def model_finetune(
        self,
        batch_size=128,
        nbits=2,
        maxsteps=500000,
        use_ib_negatives=True,
        dim=128,
        learning_rate=5e-6,
        doc_maxlen=256,
        use_relu=False,
        warmup_steps="auto",
        num_negatives=2,
    ) -> None:

        raw_data = self.train[["destination_context"]]
        raw_data.loc[:, ["passages"]] = self.train["passage_id"].map(self.passages)
        raw_data.loc[:, ["negatives"]] = self.train["passage_id"].map(
            lambda el: self.passage_sample(el, num_negatives)
        )
        raw_data = list(raw_data.itertuples(index=False, name=None))

        self.model.prepare_training_data(
            raw_data=raw_data,
            data_out_path="/cluster/scratch/mmakonnen/training-data/",
            mine_hard_negatives=False,
            num_new_negatives=0,
        )
        self.model.train(
            batch_size=batch_size,
            nbits=nbits,
            maxsteps=maxsteps,
            use_ib_negatives=use_ib_negatives,
            dim=dim,
            learning_rate=learning_rate,
            doc_maxlen=doc_maxlen,
            use_relu=use_relu,
            warmup_steps=warmup_steps,
        )

    def index(self) -> None:
        self.model.index(
            collection=list(self.passages.values()),
            document_ids=list(map(str, list(self.passages.keys()))),
            # document_metadatas=document_metadata,
            index_name="LePaRD_pretrained",
            max_document_length=256,
            split_documents=False,
        )

    def recall(self, top_k=[1, 5, 10]) -> None:
        results = self.model.search(self.test)
        # number of documents correctly classified to be in the top k results
        top_k_abs = {key: 0 for key in top_k}
        print(results)

        for i, test_search_res in enumerate(results):
            # checking if the right document has been found in the top 10
            for doc in filter(
                lambda doc: doc.document_id == self.test.passage_id.iloc[i],
                test_search_res[:k],
            ):
                for k in filter(lambda p: doc.rank < p, top_k):
                    top_k_abs[k] += 1
                break

        print(top_k)
        print([100 * top_k_abs[key] / len(self.test) for key in top_k])

    def device(self) -> str:
        if platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                return "mps"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            return "cuda"

    def passage_sample(self, id, num=10):
        """sample num passages which are not relevant to the context"""

        subset = self.train["passage_id"][self.train["passage_id"] != id]
        sample_ids = subset.sample(num)
        return list(sample_ids.map(self.passages))
