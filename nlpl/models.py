import logging
from typing import Dict, List

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


class Word2Vec(tf.keras.Model):
    def __init__(
        self,
        num_words: int,
        embedding_dim: int,
        context_size: int,
        target_size: int = 1,
    ) -> None:
        super().__init__()

        self.num_words = num_words
        self.embedding_dim = embedding_dim

        self.context_embedding = tf.keras.layers.Embedding(
            num_words,
            embedding_dim,
            input_length=context_size,
            name="context_words_embedding",
        )
        self.context_reshape = tf.keras.layers.Reshape((embedding_dim, context_size))

        self.target_embedding = tf.keras.layers.Embedding(
            num_words,
            embedding_dim,
            input_length=target_size,
            name="target_words_embedding",
        )
        self.target_reshape = tf.keras.layers.Reshape((embedding_dim, target_size))

        self.merge = tf.keras.layers.Dot(axes=1)
        self.merge_reshape = tf.keras.layers.Reshape((context_size * target_size,))

        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

        self.compile(loss="mean_squared_error", optimizer="adam")

    def call(self, inputs: List) -> tf.Tensor:
        context_input, target_input = inputs

        context_embedding = self.context_embedding(context_input)
        reshaped_context_embedding = self.context_reshape(context_embedding)

        target_embedding = self.target_embedding(target_input)
        reshaped_target_embedding = self.target_reshape(target_embedding)

        dot_product = self.merge(
            [reshaped_context_embedding, reshaped_target_embedding]
        )
        reshaped_dot_product = self.merge_reshape(dot_product)

        outputs = self.dense(reshaped_dot_product)
        return outputs

    def save_word_vectors(self, path: str, index_word: Dict[int, str]) -> None:
        weights = self.target_embedding.get_weights()[0]

        data = np.zeros((self.num_words - 1, self.embedding_dim + 1), dtype=object)
        for index, word in index_word.items():
            # Indices start from 1 because of OOV token.
            data[index - 1][0] = word
            data[index - 1][1:] = weights[index]

        np.savetxt(
            path,
            data,
            header="{} {}".format(self.num_words - 1, self.embedding_dim),
            fmt=["%s"] + ["%.12e"] * self.embedding_dim,
            delimiter=" ",
            comments="",
        )

        logger.info(f"Word vectors saved to '{path}'")
