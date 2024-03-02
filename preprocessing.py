"""Preprocessing and cleaning language dataset."""
import logging
import logging.config

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class Preprocessing:
    """Preprocessing and cleaning language dataset."""

    def __init__(self, dataset: str) -> None:
        """Initialize preprocessing data."""
        self.dataset = dataset
        self.data = {"x_valid": None, "y_valid": None, "x_train": None, "y_train": None, "x_test": None, "y_test": None}

    def process(self) -> dict:
        """Run preprocessing."""
        self.read_clean_data()
        self.split_data()

        # self.setup_training_data()

        return self.data

    def read_clean_data(self) -> None:
        """Read raw dataset and clean it for processing."""
        self.dataset = pd.read_csv(self.dataset)
        self.dataset = self.dataset.dropna()
        self.dataset["Text"] = self.dataset["Text"].astype(str)
        self.dataset["language"] = self.dataset["language"].astype(str)

        self.dataset = self.dataset.iloc[:].values

    def split_data(self) -> None:
        """Split data into Test, Train and Validation datasets."""
        xplace, self.data["x_valid"], yplace, self.data["y_valid"] = train_test_split(self.dataset[:, 0], self.dataset[:, 1], test_size=0.2, random_state=30, stratify=self.dataset[:, 1])
        self.data["x_train"], self.data["x_test"], self.data["y_train"], self.data["y_test"] = train_test_split(xplace, yplace, test_size=0.2, random_state=30, stratify=yplace)

    def categorize_rules(self) -> None:
        """Categorize languages by rules developed."""
        rules = pd.read_csv("rules.csv")

    def setup_training_data(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.data["x_train"])
        self.data["x_train"] = tokenizer.texts_to_sequences(self.data["x_train"])

        max_seq_length = max(len(seq) for seq in self.data["x_train"])
        self.data["x_train"] = tf.keras.preprocessing.sequence.pad_sequences(self.data["x_train"], maxlen=max_seq_length)


def main() -> None:
    """Main to test preprocessing."""
    preprocess = Preprocessing("dataset.csv")
    data = preprocess.process()


if __name__ == "__main__":
    main()
