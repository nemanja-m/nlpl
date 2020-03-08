from collections import Generator
from typing import List, Tuple

import numpy as np
from tensorflow.keras.preprocessing import sequence

from sampling import Sampler


class CBOWSampleGenerator(Generator):
    """Generates training samples for continous bag of words model training.

    This sample generator also supports negative sampling and batching.

    """

    def __init__(
        self,
        sequences: List[List[int]],
        sampler: Sampler,
        window_size: int = 5,
        negatives: int = 2,
        batch_size: int = 32,
    ) -> None:
        self._sequences = sequences
        self._sampler = sampler
        self._window_size = window_size
        self._negatives = negatives
        self._batch_size = batch_size

        self._reset_batch()

    def throw(self, type=None, value=None, traceback=None) -> None:
        raise super().throw(type, value, traceback)

    def send(self, _) -> Tuple[List, np.ndarray]:
        context_size: int = self._window_size * 2

        iteration = 0
        while True:
            sentence = self._sequences[iteration]
            iteration = (iteration + 1) % len(self._sequences)

            for index, word_index in enumerate(sentence):
                start = index - self._window_size
                end = index + self._window_size

                context_words: List[int] = []
                for current_word_idx in range(start, end + 1):
                    is_label_word = current_word_idx == index
                    if 0 <= current_word_idx < len(sentence) and not is_label_word:
                        context_words.append(sentence[current_word_idx])

                self._add_to_batch(
                    context_words=context_words, target_word=word_index, label=1
                )

                negative_samples = self._sampler.sample_negatives(
                    ignore_word_index=word_index, n_samples=self._negatives
                )

                for negative_word in negative_samples:
                    self._add_to_batch(
                        context_words=context_words, target_word=negative_word, label=0
                    )

                    if self._is_batch_ready():
                        return self._process_batch(context_size)

    def _is_batch_ready(self) -> bool:
        return len(self._contexts_batch) >= self._batch_size

    def _reset_batch(self) -> None:
        self._contexts_batch: List[List[int]] = []
        self._targets_batch: List[List[int]] = []
        self._labels_batch: List[int] = []

    def _add_to_batch(
        self, context_words: List[int], target_word: int, label: int
    ) -> None:
        self._contexts_batch.append(context_words)
        self._targets_batch.append([target_word])
        self._labels_batch.append(label)

    def _process_batch(self, context_size: int) -> Tuple[List, np.ndarray]:
        context_sequence = sequence.pad_sequences(
            self._contexts_batch, maxlen=context_size
        )
        target_sequence = np.array(self._targets_batch)

        # Shuffle batch
        shuffled_index = np.random.permutation(self._batch_size)
        context_sequence = context_sequence[shuffled_index]
        target_sequence = target_sequence[shuffled_index]
        labels = np.array(self._labels_batch)[shuffled_index]

        self._reset_batch()

        return [context_sequence, target_sequence], labels
