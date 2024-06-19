#!/usr/bin/env python

from pathlib import Path

from tqdm.autonotebook import tqdm
from ragatouille import RAGTrainer

from .utils import DataState

tqdm.pandas()


def gen_train_data(state: DataState) -> str:
    model = RAGTrainer(
        model_name="fine-tuned", pretrained_model_name="colbert-ir/colbertv2.0"
    )
    raw_data = state.train[["destination_context"]]
    raw_data.loc[:, ["passages"]] = state.train["passage_id"].map(state.passages)
    raw_data.loc[:, ["negatives"]] = state.train["passage_id"].progress_map(
        lambda el: passage_sample(state, el)
    )
    raw_data = list(raw_data.itertuples(index=False, name=None))

    return model.prepare_training_data(
        raw_data=raw_data,
        data_out_path=state.config["data"]["data_out_path"],
        mine_hard_negatives=False,
        num_new_negatives=0,
    )


def train_colbert(state: DataState) -> str:
    model = RAGTrainer(
        model_name="fine-tuned", pretrained_model_name="colbert-ir/colbertv2.0"
    )
    model.data_dir = Path(state.config["data"]["data_out_path"])
    return model.train(**state.config["train"])


def passage_sample(state: DataState, id) -> list:
    """sample num passages which are not relevant to the context"""

    subset = state.train["passage_id"][state.train["passage_id"] != id]
    sample_ids = subset.sample(state.config["data"]["num_negatives"])
    return list(sample_ids.map(state.passages))
