from abc import ABC, abstractmethod
from collections import Generator
from typing import List, Tuple, Union

import numpy as np
from tensorflow.keras.preprocessing import sequence

from .sampling import Sampler


# List[int] for skipgram and List[List[int]] for CBOW samples.
SamplesBatch = List[Union[int, List[int]]]


class _SampleGenerator(Generator, ABC):
    """Generates training samples for word2vec models training.

    Subclasses should also supports negative sampling and batching.

    """

    def __init__(
        self, sequences: List[List[int]], window_size: int = 5, batch_size: int = 32,
    ) -> None:
        self._sequences = sequences
        self._window_size = window_size
        self._batch_size = batch_size

        self._reset_batch()

    def throw(self, type=None, value=None, traceback=None) -> None:
        raise super().throw(type, value, traceback)

    def _is_batch_ready(self) -> bool:
        return len(self._contexts_batch) >= self._batch_size

    def _reset_batch(self) -> None:
        self._contexts_batch: SamplesBatch = []
        self._targets_batch: SamplesBatch = []
        self._labels_batch: List[int] = []

    @abstractmethod
    def _add_to_batch(
        self, context_words: List[int], target_words: List[int], labels: List[int]
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _process_batch(self) -> Tuple[List, np.ndarray]:
        raise NotImplementedError()


class CBOWSampleGenerator(_SampleGenerator):
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
        super().__init__(sequences, window_size, batch_size)

        self._sampler = sampler
        self._negatives = negatives

    def send(self, _) -> Tuple[List, np.ndarray]:
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
                    context_words=context_words, target_words=[word_index], labels=[1]
                )

                negative_samples = self._sampler.sample_negatives(
                    ignore_word_index=word_index, n_samples=self._negatives
                )

                for negative_word in negative_samples:
                    self._add_to_batch(
                        context_words=context_words,
                        target_words=[negative_word],
                        labels=[0],
                    )

                    if self._is_batch_ready():
                        return self._process_batch()

    def _add_to_batch(
        self, context_words: List[int], target_words: List[int], labels: List[int]
    ) -> None:
        self._contexts_batch.append(context_words)
        self._targets_batch.append(target_words)
        self._labels_batch.extend(labels)

    def _process_batch(self) -> Tuple[List, np.ndarray]:
        context_size = self._window_size * 2

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


class SkipgramSampleGenerator(_SampleGenerator):
    """Generates training samples for skipgram model training.

    This sample generator also supports negative sampling and batching.

    """

    def __init__(
        self,
        sequences: List[List[int]],
        num_words: int,
        window_size: int = 5,
        batch_size: int = 32,
    ):
        super().__init__(sequences, window_size, batch_size)

        self._num_words = num_words
        self._sampling_table = sequence.make_sampling_table(size=num_words)

    def send(self, _) -> Tuple[List, np.ndarray]:
        iteration = 0

        while True:
            sentence = self._sequences[iteration]
            iteration = (iteration + 1) % len(self._sequences)

            pairs, labels = sequence.skipgrams(
                sentence,
                vocabulary_size=self._num_words,
                window_size=self._window_size,
                sampling_table=self._sampling_table,
            )

            if pairs:
                target_words, context_words = [list(words) for words in zip(*pairs)]

                # Batch size is at least 32. Higher batch size is not
                # problematic.
                self._add_to_batch(context_words, target_words, labels)

                if self._is_batch_ready():
                    return self._process_batch()

    def _add_to_batch(
        self, context_words: List[int], target_words: List[int], labels: List[int]
    ) -> None:
        self._contexts_batch.extend(context_words)
        self._targets_batch.extend(target_words)
        self._labels_batch.extend(labels)

    def _process_batch(self) -> Tuple[List, np.ndarray]:
        batch = (
            [np.array(self._contexts_batch), np.array(self._targets_batch)],
            np.array(self._labels_batch),
        )

        self._reset_batch()

        return batch
