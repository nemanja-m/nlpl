import logging
import os
import pickle
from typing import List, Dict, Tuple, Iterator

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import dot, Input, Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras_preprocessing.text import Tokenizer

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
        with open(sequences_path, "rb") as fp:
            sequences = pickle.load(fp)
            logging.info(f"Precomputed text sequences loaded from '{sequences_path}'")
    else:
        logging.info("Converting text to sequences")
        sequences = tokenizer.texts_to_sequences(sentences)

        with open(sequences_path, "wb") as fp:
            pickle.dump(sequences, fp)
            logging.info(f"Sequences saved to {sequences_path}")

    return sampler, sequences


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
        pickle.dump(tokenizer, fp)
        logging.info(f"Tokenizer saved to {tokenizer_path}")

    return tokenizer


def compile_model(context_size: int, target_size: int = 1) -> Model:
    input_target = Input((target_size,))
    input_context = Input((context_size,))

    target_embedding = Embedding(
        NUM_WORDS,
        EMBEDDING_DIM,
        input_length=target_size,
        name="target_words_embedding",
    )
    target_embedding = target_embedding(input_target)
    target_embedding = Reshape((EMBEDDING_DIM, target_size))(target_embedding)

    context_embedding = Embedding(
        NUM_WORDS,
        EMBEDDING_DIM,
        input_length=context_size,
        name="context_words_embedding",
    )
    context_embedding = context_embedding(input_context)
    context_embedding = Reshape((EMBEDDING_DIM, context_size))(context_embedding)

    dot_product = dot([context_embedding, target_embedding], axes=1)
    dot_product = Reshape((context_size * target_size,))(dot_product)

    output = Dense(1, activation="sigmoid")(dot_product)

    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


if __name__ == "__main__":
    sentences_path = os.path.join(DATA_DIR, "processed", "sentences.txt")
    sampler, sequences = load_sampler_and_sequences(sentences_path)

    window_size: int = 5

    model: Model = compile_model(context_size=window_size * 2)

    sample_generator: Iterator = generators.CBOWSampleGenerator(  # type: ignore
        sequences, sampler, window_size=window_size
    )

    model.fit_generator(
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

    weights = model.layers[3].get_weights()[0][1:]
    df = pd.DataFrame(weights, index=sampler._index_word.values())

    word_vectors_path = os.path.join(ROOT_DIR, "vectors", "cbow.vec")

    np.savetxt(
        word_vectors_path,
        df.reset_index().values,
        header="{} {}".format(NUM_WORDS, EMBEDDING_DIM),
        fmt=["%s"] + ["%.12e"] * EMBEDDING_DIM,
        delimiter=" ",
        comments="",
    )

    logging.info(f"Word vectors saved to '{word_vectors_path}'")
