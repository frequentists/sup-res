#!/usr/bin/env python

import yaml, json, gzip
from typing import Dict

import pandas as pd


class DataState:
    """import context and data needed for passage retrieval"""

    def __init__(self, load_data: bool = True, load_passages: bool = True) -> None:
        self.load_yaml()
        if load_data:
            self.load_data()
        if load_passages:
            self.load_passages()

    def load_yaml(self, yaml_file_path: str = "./config/colbert.yml") -> None:
        """Load yaml file and return dictionary"""

        try:
            with open(yaml_file_path, "r") as stream:
                self.config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    def load_data(self) -> None:
        df = pd.read_csv(self.config["data"]["path"], compression="gzip")
        # df = pd.read_csv(self.config["data"]["path"], compression="gzip")
        df = df[self.config["data"]["req_rows"]]
        df = df.sample(frac=1.0, random_state=self.config["random_state"])
        df = df.reset_index(drop=True)

        self.lab2id = {int(i): j for j, i in enumerate(df.passage_id.unique())}

        train_idx = int(len(df) * self.config["data"]["train_frac"])
        dev_idx = int(
            len(df)
            * (self.config["data"]["train_frac"] + self.config["data"]["dev_frac"])
        )

        self.train = df[:train_idx]
        self.dev = df[train_idx:dev_idx]
        self.test = df[dev_idx:]

    def load_passages(self) -> Dict[int, str]:
        """load dictionary of passage_ids and passages"""

        with gzip.open(self.config["data"]["passages_path"], "rt", encoding="utf-8") as f:
            self.passages = json.load(f)
        self.passages = self.passages["data"]
        self.passages = {int(key): self.passages[key] for key in self.passages.keys()}
    
    def get_config(self):
        return self.config
