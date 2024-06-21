#!/usr/bin/env python

from ragatouille import RAGPretrainedModel

from .utils import DataState


def index_colbert(state: DataState) -> None:
    model = RAGPretrainedModel.from_pretrained(
        pretrained_model_name_or_path=state.config["index_checkpoint"]
    )
    model.index(
        collection=[state.passages[key] for key in state.lab2id.keys()],
        document_ids=[str(key) for key in state.lab2id.keys()],
        **state.config["index"]
    )
