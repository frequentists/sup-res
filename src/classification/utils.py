import pandas as pd
import random
import json
from collections import Counter

def load_data(dataframe="data/top_10000_training_data_NAACL.csv.gz"):
	df = pd.read_csv(dataframe, compression="gzip")
	df = df.sample(frac=1.0, random_state=42)
	passage2labelid = {int(i):j for j,i in enumerate(df.passage_id.unique())}

	print (len(df))
	print ("unique labels", df.passage_id.nunique())

	train_split = int(len(df) * 0.9)
	dev_split = int(len(df) * 0.95)
	train = df.iloc[:train_split]
	dev = df.iloc[train_split:dev_split]
	test = df.iloc[dev_split:]
	print (len(train), len(dev), len(test))
	return train, dev, test, passage2labelid


# train, dev, test = load_data()

def load_passages(fn = "unique_citations_origin_only_valid_passages.jsonl"):
	with open(fn) as f:
		data = [json.loads(i) for i in f]
	return data
	
def eval(query, predictions):
	#TODO implement
	pass

