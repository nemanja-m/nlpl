import logging
import os
import re
from typing import Tuple, Optional

import click
import pandas as pd


logger = logging.getLogger(__name__)

DATA_DIR: str = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
DEFAULT_RAW_PATH: str = os.path.join(RAW_DATA_DIR, "dataset.csv")

PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
DEFAULT_PROCESSED_PATH: str = os.path.join(PROCESSED_DATA_DIR, "dataset.csv")

START_TOKEN: str = "<start>"
END_TOKEN: str = "<end>"


def _preprocess_line(line: str) -> Optional[str]:
    line = line.lower().strip()

    # Creating a space between a word and the punctuation following it.
    # Example: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    line = re.sub(r"([?.!,])", r" \1 ", line)
    line = re.sub(r'[" "]+', " ", line)

    line = re.sub(r"[^a-zA-Z?.!,]+", " ", line).strip()

    if not line:
        return pd.NA

    # Start and end token prepares line for sequence to sequence models.
    line = f"{START_TOKEN} {line} {END_TOKEN}"

    return line


def preprocess_row(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    processed_input = _preprocess_line(row.input)
    processed_target = _preprocess_line(row.target)
    return processed_input, processed_target


@click.command(help="Preprocess dataset for chatbot sequence to sequence modeling.")
@click.option(
    "--source",
    "-s",
    type=str,
    default=DEFAULT_RAW_PATH,
    help="Path to the raw input CSV dataset.",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    default=DEFAULT_PROCESSED_PATH,
    help="Path to the processed output CSV dataset.",
)
def preprocess(source: str, destination: str) -> None:
    logging.info(f"Preprocessing raw dataset from '{source}'")

    # Dataset is CSV file with 'input' and 'target' columns.
    df = pd.read_csv(source)

    processed_df = df.apply(preprocess_row, axis=1, result_type="broadcast")
    processed_df.dropna(inplace=True)

    processed_df.to_csv(destination, index=False)

    logging.info(f"Preprocessed dataset saved to '{destination}'")


if __name__ == "__main__":
    preprocess()
