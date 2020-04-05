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


class Attention(tf.keras.layers.Layer):
    """Additive style attention layer."""

    def __init__(self, units: int) -> None:
        super().__init__()

        self.query_weights = tf.keras.layers.Dense(units)
        self.value_weights = tf.keras.layers.Dense(units)
        self.score_weights = tf.keras.layers.Dense(1)

    def call(self, query: tf.Tensor, values: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Transform input query shape from [batch_size, hidden_dim] to
        # [batch_size, 1, hidden_dim]. This is required to calculate attention
        # score.
        query = tf.expand_dims(query, 1)

        tanh_values = tf.nn.tanh(self.query_weights(query) + self.value_weights(values))
        score = self.score_weights(tanh_values)

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int, decoding_units: int):
        super().__init__()

        # TODO: Initialize embeddings with pretrained weights.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim
        )
        self.gru = tf.keras.layers.GRU(
            units=decoding_units,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )
        self.dense = tf.keras.layers.Dense(units=vocab_size)
        self.attention = Attention(units=decoding_units)

    def call(self, inputs, hidden_state, encoder_output):
        context_vector, attention_weights = self.attention(hidden_state, encoder_output)

        embeddings = self.embedding(inputs)

        # GRU input tensor of [batch_size, 1, embedding_dim + decoding_units].
        gru_input = tf.concat([tf.expand_dims(context_vector, 1), embeddings], axis=-1)
        output, state = self.gru(gru_input)

        # Convert GRU output tensor from [batch_size, 1, decoding_units] to
        # [batch_size, decoding_units] shape.
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.dense(output)

        return output, state, attention_weights
