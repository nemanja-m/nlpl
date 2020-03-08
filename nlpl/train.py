import os
from typing import Iterator

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model

from . import generators, models, paths
from .loaders import SequenceLoader


NUM_WORDS: int = 50_000 + 1  # +1 for PAD token.
EMBEDDING_DIM: int = 256

EPOCHS: int = 20
STEPS_PER_EPOCH: int = 10_000_000


if __name__ == "__main__":
    sentences_path: str = os.path.join(paths.DATA_DIR, "processed", "sentences.txt")
    loader: SequenceLoader = SequenceLoader(sentences_path, num_words=NUM_WORDS)

    window_size: int = 5

    model: Model = models.Word2Vec(
        num_words=NUM_WORDS, embedding_dim=EMBEDDING_DIM, context_size=window_size * 2
    )
    model.compile(loss="mean_squared_error", optimizer="adam")

    sample_generator: Iterator = generators.CBOWSampleGenerator(  # type: ignore
        loader.sequences, loader.sampler, window_size=window_size
    )

    model.fit(
        sample_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[
            TensorBoard(
                log_dir=paths.TENSORBOARD_LOGS_DIR,
                write_grads=False,
                write_graph=False,
                histogram_freq=0,
            ),
            ModelCheckpoint(
                filepath=os.path.join(paths.MODELS_DIR, "weights.{epoch:02d}.hdf5"),
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ],
    )

    word_vectors_path: str = os.path.join(paths.VECTORS_DIR, "cbow.vec")
    model.save_word_vectors(
        path=word_vectors_path, index_word=loader.sampler.index_word
    )
