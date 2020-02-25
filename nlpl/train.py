import logging
import os
import pickle
from typing import List

from keras_preprocessing.text import Tokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


current_file: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(current_file)
DATA_DIR: str = os.path.join(ROOT_DIR, "data")
CACHE_DIR: str = os.path.join(ROOT_DIR, ".cache")

NUM_WORDS: int = 100_000


def read_sentences(path: str) -> List[str]:
    with open(path, "r") as fp:
        sentences: List[str] = fp.read().splitlines()

    logging.info(f"Data loaded from {path}")

    return sentences


def create_tokenizer(
    sentences: List[str], num_words: int = NUM_WORDS, use_cached: bool = True
) -> Tokenizer:
    tokenizer_path: str = os.path.join(CACHE_DIR, "tokenizer.pkl")

    if os.path.exists(tokenizer_path) and use_cached:
        logging.info(f"Loading cached tokenizer from {tokenizer_path}")

        with open(tokenizer_path, "rb") as fp:
            cached_tokenizer: Tokenizer = pickle.load(fp)
            return cached_tokenizer

    tokenizer: Tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(sentences)

    logging.info(f"Tokenizer fit on text")

    if not os.path.exists(CACHE_DIR):
        logging.info(f"Cache folder does not exist. Creating '{CACHE_DIR}' folder")
        os.mkdir(CACHE_DIR)

    with open(tokenizer_path, "wb") as fp:
        logging.info(f"Tokenizer saved to {tokenizer_path}")
        pickle.dump(tokenizer, fp)

    return tokenizer


if __name__ == "__main__":
    sentences_path: str = os.path.join(DATA_DIR, "processed", "sentences.txt")
    sentences: List[str] = read_sentences(sentences_path)

    tokenizer: Tokenizer = create_tokenizer(sentences)
