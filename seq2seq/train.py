import os
import time
from typing import List, Tuple

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from model import Encoder, Decoder


current_dir: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(current_dir, "data")

PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
DEFAULT_PROCESSED_PATH: str = os.path.join(PROCESSED_DATA_DIR, "dataset.csv")

BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_SIZE = 1024
EPOCHS = 20


def load_dataset() -> Tuple[List[str], List[str]]:
    df = pd.read_csv(DEFAULT_PROCESSED_PATH)
    return df.input.tolist(), df.target.tolist()


def tokenize(texts: List[str]) -> Tuple:
    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding="post")

    return tokenizer, padded_sequences


def calculate_loss(loss_object, true, pred):
    mask = tf.math.logical_not(tf.math.equal(true, 0))  # Mask OOV tokens.
    loss = loss_object(true, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)


def train(
    batch_size: int = BATCH_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    units: int = HIDDEN_SIZE,
    epochs: int = EPOCHS,
):
    input_texts, target_texts = load_dataset()

    input_tokenizer, input_sequences = tokenize(texts=input_texts)
    target_tokenizer, target_sequences = tokenize(texts=target_texts)

    buffer_size = input_sequences.shape[0]
    steps_per_epoch = buffer_size // batch_size

    input_vocab_size = len(input_tokenizer.word_index) + 1  # +1 for OOV token.
    target_vocab_size = len(target_tokenizer.word_index) + 1  # +1 for OOV token.

    dataset = (
        tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))
        .shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
    )

    ex_in_batch, ex_tar_batch = next(iter(dataset))

    encoder = Encoder(
        vocab_size=input_vocab_size,
        embedding_dim=embedding_dim,
        encoding_units=units,
        batch_size=batch_size,
    )

    decoder = Decoder(
        vocab_size=target_vocab_size, embedding_dim=embedding_dim, decoding_units=units,
    )

    optimizer = tf.keras.optimizers.Adam()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    @tf.function
    def train_step(input_batch, target_batch):
        loss = 0

        with tf.GradientTape() as tape:
            encoder_output, encoder_hidden_state = encoder(input_batch)

            decoder_hidden_state = encoder_hidden_state
            decoder_input = tf.expand_dims(
                [target_tokenizer.word_index["<start>"]] * batch_size, 1
            )

            target_sequence_length = target_batch.shape[1]

            for i in range(1, target_sequence_length):
                predictions, decoder_hidden_state, _ = decoder(
                    decoder_input, decoder_hidden_state, encoder_output
                )

                current_token = target_batch[:, i]
                loss += calculate_loss(loss_object, current_token, predictions)

                # Teacher forcing: next decoder input is current token.
                decoder_input = tf.expand_dims(current_token, 1)

        batch_loss = loss / int(target_sequence_length)

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    for epoch in range(epochs):
        start = time.time()

        total_loss = 0

        for (batch_number, (input_batch, target_batch)) in enumerate(
            dataset.take(steps_per_epoch)
        ):
            batch_loss = train_step(input_batch, target_batch)
            total_loss += batch_loss

            if batch_number % 100 == 0:
                print(
                    "Epoch {} Batch {} Loss {:.4f}".format(
                        epoch + 1, batch_number, batch_loss.numpy()
                    )
                )

        if (epoch + 1) % 2 == 0:
            encoder.save_weights(f"./models/encoder.{epoch}", save_format="tf")
            decoder.save_weights(f"./models/decoder.{epoch}", save_format="tf")

        print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))
        print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    print("\nTraining done. Evaluating...")
    evaluate(encoder, decoder)


def evaluate(encoder=None, decoder=None):
    input_texts, target_texts = load_dataset()

    input_tokenizer, input_sequences = tokenize(texts=input_texts)
    target_tokenizer, target_sequences = tokenize(texts=target_texts)

    input_vocab_size = len(input_tokenizer.word_index) + 1  # +1 for OOV token.
    target_vocab_size = len(target_tokenizer.word_index) + 1  # +1 for OOV token.
    max_length_targ = target_sequences.shape[1]

    if encoder is None:
        encoder = Encoder(
            vocab_size=input_vocab_size,
            embedding_dim=EMBEDDING_DIM,
            encoding_units=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
        )
        encoder.load_weights("./models/encoder.1")

    if decoder is None:
        decoder = Decoder(
            vocab_size=target_vocab_size,
            embedding_dim=EMBEDDING_DIM,
            decoding_units=HIDDEN_SIZE,
        )
        decoder.load_weights("./models/decoder.1")

    while True:
        try:
            query = input(">")

            tokens = [
                input_tokenizer.word_index[w] if w in input_tokenizer.word_index else 0
                for w in query.split()
            ]
            padded_tokens = pad_sequences(
                [tokens], maxlen=max_length_targ, padding="post"
            )

            query_tensor = tf.convert_to_tensor(padded_tokens)

            result = ""
            hidden = [tf.zeros((1, HIDDEN_SIZE))]

            enc_out, enc_hidden = encoder(query_tensor, hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]], 0)

            for t in range(max_length_targ):
                predictions, dec_hidden, attention_weights = decoder(
                    dec_input, dec_hidden, enc_out
                )

                predicted_id = tf.argmax(predictions[0]).numpy()
                predicted_token = target_tokenizer.index_word[predicted_id]

                if predicted_token == "<end>":
                    break

                result += predicted_token + " "

                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)

            print(result.strip())
            print()

        except (KeyboardInterrupt, EOFError):
            return


if __name__ == "__main__":
    train()
    # evaluate()
