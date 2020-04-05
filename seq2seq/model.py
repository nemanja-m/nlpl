from typing import Tuple

import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(
        self, vocab_size: int, embedding_dim: int, encoding_units: int, batch_size: int
    ):
        super().__init__()

        self.encoding_units = encoding_units
        self.batch_size = batch_size

        # TODO: Initialize embeddings with pretrained weights.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim
        )
        self.gru = tf.keras.layers.GRU(
            units=encoding_units,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(
        self, input_batch: tf.Tensor, hidden_state: tf.Tensor = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if hidden_state is None:
            hidden_state = self._init_hidden_state()

        embeddings = self.embedding(input_batch)
        output, final_state = self.gru(embeddings, initial_state=hidden_state)
        return output, final_state

    def _init_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoding_units))
