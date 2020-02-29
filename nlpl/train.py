import logging
import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
from keras.models import Model
from keras.layers import dot, Input, Dense, Reshape
from keras.layers.embeddings import Embedding
from keras_preprocessing.text import Tokenizer
from keras_preprocessing import sequence

from sampling import Sampler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


current_file: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(current_file)
DATA_DIR: str = os.path.join(ROOT_DIR, "data")
CACHE_DIR: str = os.path.join(ROOT_DIR, ".cache")

NUM_WORDS: int = 100_000
EMBEDDING_DIM: int = 300


def load_sampler_and_sequences(sentences_path: str) -> Tuple[Sampler, List[List[int]]]:
    sentences: List[str] = read_sentences(sentences_path)

    tokenizer: Tokenizer = create_tokenizer(sentences)

    # Limit number of words from corpus used in training.
    word_counts: Dict[str, int] = {}
    index_word: Dict[int, str] = {}
    for word, index in tokenizer.word_index.items():
        if index <= NUM_WORDS:
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


def generate_training_samples(
    sequences: List[List[int]],
    sampler: Sampler,
    window_size: int = 5,
    batch_size: int = 32,
):
    context_size: int = window_size * 2

    samples_batch = []
    labels_batch = []

    total_batches = 0

    i = 0
    while True:

        if i == (len(sequences) - 1):
            print(f"\n\ntotal batches: {total_batches}\n\n")
            break

        sentence = sequences[i]
        i = (i + 1) % len(sequences)

        for index, word_index in enumerate(sentence):
            start = index - window_size
            end = index + window_size

            context_words: List[int] = []
            for current_word_idx in range(start, end + 1):
                is_label_word = current_word_idx == index
                if 0 <= current_word_idx < len(sentence) and not is_label_word:
                    context_words.append(sentence[current_word_idx])

            samples_batch.append(context_words)
            labels_batch.append([word_index])

            if len(samples_batch) == batch_size:
                samples_sequence = sequence.pad_sequences(
                    samples_batch, maxlen=context_size
                )
                labels_sequence = sequence.pad_sequences(
                    labels_batch, maxlen=context_size
                )
                samples_batch = []
                labels_batch = []

                total_batches += 1

                yield [samples_sequence, labels_sequence], np.ones((batch_size,))


def compile_model() -> Model:
    input_size = 10
    input_target = Input((input_size,))
    input_context = Input((input_size,))

    embedding = Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=input_size)

    word_embedding = embedding(input_target)
    word_embedding = Reshape((EMBEDDING_DIM, input_size))(word_embedding)

    context_embedding = embedding(input_context)
    context_embedding = Reshape((EMBEDDING_DIM, input_size))(context_embedding)

    dot_product = dot([word_embedding, context_embedding], axes=1)
    dot_product = Reshape((input_size ** 2,))(dot_product)

    output = Dense(1, activation="sigmoid")(dot_product)

    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model


if __name__ == "__main__":
    sentences_path: str = os.path.join(DATA_DIR, "processed", "sentences.txt")
    sampler, sequences = load_sampler_and_sequences(sentences_path)

    model = compile_model()
    model.fit_generator(
        generate_training_samples(sequences, sampler), steps_per_epoch=128, epochs=10
    )
