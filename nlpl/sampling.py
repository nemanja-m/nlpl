from typing import Dict

import numpy as np


class Sampler:
    def __init__(self, word_counts: Dict[str, int], sample_rate: float = 1e-3):
        self.sample_rate = sample_rate
        self._total_words: int = sum(word_counts.values())
        self._word_counts: Dict[str, int] = word_counts

    def should_keep_word(self, word: str) -> bool:
        if word not in self._word_counts:
            return False

        word_count: int = self._word_counts[word]
        word_fraction: float = word_count / self._total_words

        prob: float = (np.sqrt(word_fraction / self.sample_rate) + 1) * (
            self.sample_rate / word_fraction
        )

        return np.random.random() <= prob
