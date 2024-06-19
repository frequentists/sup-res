#!/usr/bin/env python

from ragatouille import RAGPretrainedModel

from .utils import DataState


def index_colbert(state: DataState) -> None:
    model = RAGPretrainedModel.from_pretrained(
        pretrained_model_name_or_path=state.config["index_checkpoint"]
    )
    model.index(
        collection=list(state.passages.values()),
        document_ids=list(map(str, list(state.passages.keys()))),
        **state.config["index"]
    )
