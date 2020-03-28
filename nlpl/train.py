import os
from typing import Iterator

import click
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model

from . import generators, models, paths
from .loaders import SequenceLoader


SENTENCES_PATH: str = os.path.join(paths.DATA_DIR, "processed", "sentences.txt")

NUM_WORDS: int = 50_000 + 1  # +1 for PAD token.
EMBEDDING_DIM: int = 300

EPOCHS: int = 20
STEPS_PER_EPOCH: int = 10_000_000
BATCH_SIZE: int = 256


@click.command(
    help="Train word2vec models using skipgram or CBOW with negative sampling."
)
@click.option(
    "--sentences-path",
    default=SENTENCES_PATH,
    type=str,
    help="Path to file with training corpus/sentences. Each line should be one sentence.",
)
@click.option(
    "--algorithm",
    default="skipgram",
    type=str,
    show_default=True,
    help="Training algorithm. Supported values are: 'skipgram' and 'cbow'.",
)
@click.option(
    "--num-words",
    default=NUM_WORDS,
    type=int,
    show_default=True,
    help="Number of the most frequent words from corpus that are used in training.",
)
@click.option(
    "--window-size",
    default=10,
    type=int,
    show_default=True,
    help="Size of context window.",
)
@click.option(
    "--embedding-dim",
    default=EMBEDDING_DIM,
    type=int,
    show_default=True,
    help="Dimension of word vectors.",
)
@click.option(
    "--epochs",
    default=EPOCHS,
    type=int,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--steps",
    default=STEPS_PER_EPOCH,
    type=int,
    show_default=True,
    help="Steps per one epoch.",
)
@click.option(
    "--batch-size",
    default=BATCH_SIZE,
    type=int,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--tensorboard", is_flag=True, help="Use tensorboard logging.",
)
def train(
    sentences_path: str,
    algorithm: str,
    num_words: int,
    window_size: int,
    embedding_dim: int,
    epochs: int,
    steps: int,
    batch_size: int,
    tensorboard: bool,
):
    loader: SequenceLoader = SequenceLoader(sentences_path, num_words=num_words)

    context_size = 1 if algorithm.lower() == "skipgram" else window_size * 2

    model: Model = models.Word2Vec(
        num_words=num_words, embedding_dim=embedding_dim, context_size=context_size
    )

    sample_generator: Iterator
    if algorithm.lower() == "skipgram":
        sample_generator = generators.SkipgramSampleGenerator(  # type: ignore
            sequences=loader.sequences,
            num_words=num_words,
            window_size=window_size,
            batch_size=batch_size,
        )
    else:
        sample_generator = generators.CBOWSampleGenerator(  # type: ignore
            loader.sequences,
            loader.sampler,
            window_size=window_size,
            batch_size=batch_size,
        )

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(
                paths.MODELS_DIR, algorithm + ".weights.{epoch:02d}.hdf5"
            ),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    if tensorboard:
        tensorboard_callback = TensorBoard(
            log_dir=paths.TENSORBOARD_LOGS_DIR,
            write_grads=False,
            write_graph=False,
            histogram_freq=0,
        )
        callbacks.append(tensorboard_callback)

    model.fit(
        sample_generator, steps_per_epoch=steps, epochs=epochs, callbacks=callbacks
    )

    word_vectors_path: str = os.path.join(
        paths.VECTORS_DIR, "{model}.vec".format(model=algorithm)
    )

    model.save_word_vectors(
        path=word_vectors_path, index_word=loader.sampler.index_word
    )


if __name__ == "__main__":
    train()
