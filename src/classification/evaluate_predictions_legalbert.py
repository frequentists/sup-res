import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import ndcg_score
from utils import *
import json
import pytrec_eval
import json


def convert_to_trec_eval(true, hits):
    qrel = {}
    run = {}
    for i, ref in enumerate(true):
        key = "q" + str(i)
        qrel[key] = {"d" + str(ref): 1}
    for i, pred in enumerate(hits):
        key = "q" + str(i)
        run[key] = {"d" + str(p): 10 - j for j, p in enumerate(pred)}
    return qrel, run


for n_labels in ["10000", "20000", "50000"]:
    # ...

    # train, dev, test, passage2labelid = load_data(dataframe = "../data/top_" + n_labels + "_training_data_NAACL.csv.gz")
    # fn = "classification-predictions/naacl-finetuned-distilbert-base-uncased-" + n_labels + "/predictions_devset_" + n_labels + ".json"
    fn = (
        "classification-predictions/legal-bert-base-uncased-"
        + n_labels
        + "/predictions_devset_"
        + n_labels
        + ".json"
    )
    with open(fn) as f:
        data = json.load(f)

    true = data["true"]
    predicted = data["hits"]
    hits = predicted
    for recall_at in [1, 5, 10]:
        is_correct = []
        for i, t in enumerate(true):
            predictions = predicted[i][:recall_at]
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

    qrel, run = convert_to_trec_eval(true, hits)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"map", "ndcg"})
    results = evaluator.evaluate(run)
    print(
        "devset ndcg", np.round(np.mean([i["ndcg"] for i in results.values()]), 4) * 100
    )
    print(
        "devset map", np.round(np.mean([i["map"] for i in results.values()]), 4) * 100
    )

    fn = (
        "classification-predictions/legal-bert-base-uncased-"
        + n_labels
        + "/predictions_testset_"
        + n_labels
        + ".json"
    )
    with open(fn) as f:
        data = json.load(f)

    true = data["true"]
    predicted = data["hits"]
    hits = predicted
    for recall_at in [1, 5, 10]:
        is_correct = []
        for i, t in enumerate(true):
            predictions = predicted[i][:recall_at]
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

    qrel, run = convert_to_trec_eval(true, hits)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"map", "ndcg"})
    results = evaluator.evaluate(run)
    print(
        "testset ndcg",
        np.round(np.mean([i["ndcg"] for i in results.values()]), 4) * 100,
    )
    print(
        "testset map", np.round(np.mean([i["map"] for i in results.values()]), 4) * 100
    )
