import platform
import yaml, json
import torch
import numpy as np
import pandas as pd
from csv import DictReader
from ragatouille import RAGPretrainedModel, RAGTrainer
from sklearn.model_selection import StratifiedKFold

import sys
print(sys.version)

# sbatch --time=08:00 --mem-per-cpu=12G --gpus=1 --gres=gpumem:10G --mail-type=END --mail-user="" --wrap="python scripts.py"