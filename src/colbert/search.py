#!/usr/bin/env python

from tqdm import tqdm
from ragatouille import RAGPretrainedModel

from .utils import DataState


def search_colbert(state: DataState) -> None:
    model = RAGPretrainedModel.from_index(index_path=state.config["search_checkpoint"])
    results = model.search(
        state.test.destination_context.values,
        k=state.config["search"]["initial"] if state.config["search"]["rerank"] else 10,
    )
    # number of documents correctly classified to be in the top k results
    top_k_abs = {key: 0 for key in state.config["search"]["top_k"]}

    for i, test_search_res in tqdm(enumerate(results), total=len(results)):
        refined_search_res = model.rerank(
            query=state.test.destination_context.values[i],
            documents=[el["content"] for el in test_search_res],
        )

        # checking if the right document has been found in the top 10
        for doc in filter(
            lambda doc: test_search_res[doc["result_index"]]["document_id"] == str(state.test.passage_id.iloc[i]),
            refined_search_res,
        ):
            for k in filter(lambda p: doc["rank"] < p, state.config["search"]["top_k"]):
                top_k_abs[k] += 1
            break

    print(state.config["search"]["top_k"])
    print(
        [
            100 * top_k_abs[key] / len(state.test)
            for key in state.config["search"]["top_k"]
        ]
    )
