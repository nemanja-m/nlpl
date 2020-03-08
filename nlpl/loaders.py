import gc
import logging
import os
import pickle
from typing import List, Dict, Optional

from tensorflow.keras.preprocessing.text import Tokenizer

import paths
from sampling import Sampler


class SequenceLoader:
    def __init__(self, sentences_path: str, num_words: int) -> None:
        self._sentences_path = sentences_path
        self._num_words = num_words

        self._sentences: List[str] = []
        self._tokenizer: Optional[Tokenizer] = None
        self._sequences: List[List[int]] = []
        self._sampler: Optional[Sampler] = None

    @property
    def sentences(self) -> List[str]:
        if not self._sentences:
            self._load_sentences()
        return self._sentences

    def _load_sentences(self) -> None:
        with open(self._sentences_path, "r") as fp:
            self._sentences = fp.read().splitlines()
        logging.info(f"Sentences loaded from {self._sentences_path}")

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._load_tokenizer()

        assert self._tokenizer is not None

        return self._tokenizer

    def _load_tokenizer(self) -> None:
        tokenizer_path: str = os.path.join(paths.CACHE_DIR, "tokenizer.pkl")

        tokenizer: Tokenizer
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as fp:
                tokenizer = pickle.load(fp)
                logging.info(f"Tokenizer loaded from {tokenizer_path}")
        else:
            tokenizer = Tokenizer(num_words=self._num_words)
            tokenizer.fit_on_texts(self.sentences)

            logging.info(f"Tokenizer fit on texts")

            if not os.path.exists(paths.CACHE_DIR):
                logging.info(
                    f"Cache folder does not exist. Creating '{paths.CACHE_DIR}' folder"
                )
                os.mkdir(paths.CACHE_DIR)

            with open(tokenizer_path, "wb") as fp:
                pickle.dump(tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"Tokenizer saved to {tokenizer_path}")

        self._tokenizer = tokenizer

    @property
    def sequences(self):
        if not self._sequences:
            self._load_sequences()
        return self._sequences

    def _load_sequences(self) -> None:
        sequences_path: str = os.path.join(paths.DATA_DIR, "processed", "sequences.pkl")

        if os.path.exists(sequences_path):
            sequences = _load_sequences(sequences_path)
            logging.info(f"Cached text sequences loaded from '{sequences_path}'")
        else:
            logging.info("Converting sentences to sequences")
            sequences = self.tokenizer.texts_to_sequences(self.sentences)

            with open(sequences_path, "wb") as fp:
                pickle.dump(sequences, fp, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"Sequences saved to '{sequences_path}'")

        self._sequences = sequences

    @property
    def sampler(self) -> Sampler:
        if self._sampler is None:
            self._load_sampler()

        assert self._sampler is not None

        return self._sampler

    def _load_sampler(self) -> None:
        # Limit number of words from corpus used in training.
        word_counts: Dict[str, int] = {}
        index_word: Dict[int, str] = {}

        for word, index in self.tokenizer.word_index.items():
            if 0 < index < self._num_words:
                word_counts[word] = self.tokenizer.word_counts[word]
                index_word[index] = word

        self._sampler = Sampler(word_counts=word_counts, index_word=index_word)
        logging.info("Word sampler initialized")


def _load_sequences(path: str) -> List[List[int]]:
    sequences: List[List[int]]
    with open(path, "rb") as fp:
        # Disabling garbage collector before loading sequences greatly speeds
        # up loading time.
        gc.disable()

        sequences = pickle.load(fp)

        # Garbage collector must be enabled again.
        gc.enable()

    return sequences
