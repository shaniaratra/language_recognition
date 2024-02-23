"""Preprocessing and cleaning language dataset."""
import logging
import logging.config

import pandas as pd

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class Preprocessing:
    """Preprocessing and cleaning language dataset."""

    def __init__(self, dataset: str) -> None:
        """Initialize preprocessing data."""
        self.dataset = dataset
        self.data = None

    def process(self) -> None:
        """Run preprocessing."""
        self.read_clean_data()

    def read_clean_data(self) -> None:
        """Read raw dataset and clean it for processing."""
        self.data = pd.read_csv(self.dataset)
        self.data = self.data.dropna()
        self.data["Text"] = self.data["Text"].astype(str)
        self.data["language"] = self.data["language"].astype(str)

    def categorize_rules(self) -> None:
        """Categorize languages by rules developed."""
        rules = pd.read_csv("rules.csv")
        # TODO parse data against rules


def main() -> None:
    """Main to test preprocessing."""
    preprocess = Preprocessing("dataset.csv")
    preprocess.process()


if __name__ == "__main__":
    main()
