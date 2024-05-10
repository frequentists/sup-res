from utils import *
for n_labels in ["10000", "20000", "50000"]:
	train, dev, test, passage2labelid = load_data(f"data/top_${n_labels}_training_data_NAACL.csv.gz")
	print (n_labels, len(train), len(dev), len(test))
