import gc
import logging
import os
import pickle
from typing import List, Dict, Tuple, Iterator

import models
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer

import generators
from sampling import Sampler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


current_file: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(current_file)
DATA_DIR: str = os.path.join(ROOT_DIR, "data")
CACHE_DIR: str = os.path.join(ROOT_DIR, ".cache")
MODELS_DIR: str = os.path.join(ROOT_DIR, "models")

NUM_WORDS: int = 50_000 + 1  # +1 for PAD token.
EMBEDDING_DIM: int = 256

EPOCHS: int = 20
STEPS_PER_EPOCH: int = 10_000_000

TENSORBOARD_LOGS_DIR: str = os.path.join(ROOT_DIR, ".tensorboard_logs")


def load_sampler_and_sequences(sentences_path: str) -> Tuple[Sampler, List[List[int]]]:
    sentences: List[str] = read_sentences(sentences_path)

    tokenizer: Tokenizer = create_tokenizer(sentences)

    # Limit number of words from corpus used in training.
    word_counts: Dict[str, int] = {}
    index_word: Dict[int, str] = {}
    for word, index in tokenizer.word_index.items():
        if 0 < index < NUM_WORDS:
            word_counts[word] = tokenizer.word_counts[word]
            index_word[index] = word

    sampler = Sampler(word_counts=word_counts, index_word=index_word)
    logging.info("Sampler initialized")

    sequences_path: str = os.path.join(DATA_DIR, "processed", "sequences.pkl")
    if os.path.exists(sequences_path):
        sequences = load_sequences(sequences_path)
    else:
        logging.info("Converting text to sequences")
        sequences = tokenizer.texts_to_sequences(sentences)

        with open(sequences_path, "wb") as fp:
            pickle.dump(sequences, fp, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Sequences saved to {sequences_path}")

    return sampler, sequences


def load_sequences(sequences_path: str) -> List[List[int]]:
    sequences: List[List[int]]
    with open(sequences_path, "rb") as fp:
        # Disabling garbage collector before loading sequences greatly speeds
        # up loading time.
        gc.disable()

        sequences = pickle.load(fp)

        # Garbage collector must be enabled again.
        gc.enable()

    logging.info(f"Precomputed text sequences loaded from '{sequences_path}'")
    return sequences


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
        with open(tokenizer_path, "rb") as fp:
            cached_tokenizer: Tokenizer = pickle.load(fp)
            logging.info(f"Tokenizer loaded from {tokenizer_path}")
            return cached_tokenizer

    tokenizer: Tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(sentences)

    logging.info(f"Tokenizer fit on texts")

    if not os.path.exists(CACHE_DIR):
        logging.info(f"Cache folder does not exist. Creating '{CACHE_DIR}' folder")
        os.mkdir(CACHE_DIR)

    with open(tokenizer_path, "wb") as fp:
        pickle.dump(tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Tokenizer saved to {tokenizer_path}")

    return tokenizer


if __name__ == "__main__":
    sentences_path = os.path.join(DATA_DIR, "processed", "sentences.txt")
    sampler, sequences = load_sampler_and_sequences(sentences_path)

    window_size: int = 5

    model: Model = models.Word2Vec(
        num_words=NUM_WORDS, embedding_dim=EMBEDDING_DIM, context_size=window_size * 2
    )
    model.compile(loss="mean_squared_error", optimizer="adam")

    sample_generator: Iterator = generators.CBOWSampleGenerator(  # type: ignore
        sequences, sampler, window_size=window_size
    )

    model.fit(
        sample_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[
            TensorBoard(
                log_dir=TENSORBOARD_LOGS_DIR,
                write_grads=False,
                write_graph=False,
                histogram_freq=0,
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, "weights.{epoch:02d}.hdf5"),
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ],
    )

    word_vectors_path = os.path.join(ROOT_DIR, "vectors", "cbow.vec")
    model.save_word_vectors(path=word_vectors_path, index_word=sampler._index_word)
    logging.info(f"Word vectors saved to '{word_vectors_path}'")
