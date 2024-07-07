#!/usr/bin/env python

from tqdm import tqdm
from ragatouille import RAGPretrainedModel
from .utils import DataState


# Function to normalize scores
def normalize_scores(scores):
    max_score = max(scores)
    min_score = min(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def scores_colbert(state: DataState, batch, n, norm_func) -> None:

    model = RAGPretrainedModel.from_index(index_path=state.config["search_checkpoint"])
    results = model.search(
        batch,
        k=state.config["search"]["initial"] if state.config["search"]["rerank"] else 10,
    )

    batch_vectors = []  # To store the vectors for each sample in the batch

    for i, test_search_res in tqdm(enumerate(results), total=len(results)):
        refined_search_res = model.rerank(
            query=batch[i],
            documents=[el["content"] for el in test_search_res],
        )

        norm_scores = norm_func([el["score"] for el in refined_search_res])

        vector = [0] * n  # Initialize vector with zeros
        for i, doc in enumerate(refined_search_res):
            label = state.lab2id[test_search_res[doc["result_index"]]["document_id"]]
            vector[label] = norm_scores[i]
            
        batch_vectors.append(vector)