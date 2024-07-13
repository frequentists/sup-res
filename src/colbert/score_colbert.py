#!/usr/bin/env python

from tqdm import tqdm
from ragatouille import RAGPretrainedModel
import torch

from .utils import DataState

class ScoreColBERT:
    def __init__(self, state: DataState):
        self.state = state
        self.model = RAGPretrainedModel.from_index(index_path=self.state.config["search_checkpoint"])

    # Static method to normalize scores
    @staticmethod
    def normalize_scores(scores):
        #max_score = max(scores)
        #min_score = min(scores)
        return scores

    def scores_colbert(self, batch, n, norm_func=None):
        if norm_func is None:
            norm_func = self.normalize_scores

        results = self.model.search(
            batch,
            k=self.state.config["search"]["initial"] if self.state.config["search"]["rerank"] else 10,
        )

        batch_vectors = []  # To store the pythonvectors for each sample in the batch

        for i, test_search_res in enumerate(results):
            # print(test_search_res)
            refined_search_res = self.model.rerank(
                query=batch[i],
                documents=[el["content"] for el in test_search_res],
            )

            norm_scores = [el["score"] for el in refined_search_res]

            vector = [0] * n  # Initialize vector with zeros
            for j, doc in enumerate(refined_search_res):
                label = self.state.lab2id[int(test_search_res[doc["result_index"]]["document_id"])]
                vector[label] = norm_scores[j]

            batch_vectors.append(vector)
        return batch_vectors

# Example usage:
# state = DataState(...)  # Initialize the DataState object with appropriate arguments
# scorer = ScoreColBERT(state)
# batch_vectors = scorer.scores_colbert(batch, n)
