from typing import Dict, Tuple, List

import numpy as np


class Sampler:
    def __init__(
        self,
        word_counts: Dict[str, int],
        sample_rate: float = 1e-3,
        word_prob_power: float = 0.75,
    ):
        self.sample_rate: float = sample_rate

        self._total_words, self._pow_total_words = self._count_total_words(
            word_counts, word_prob_power
        )
        self._word_probs, self._pow_word_probs = self._calculate_word_probabilities(
            word_counts, word_prob_power
        )

    def _count_total_words(
        self, word_counts: Dict[str, int], word_prob_power: float
    ) -> Tuple[int, float]:
        total_words: int = 0
        pow_total_words: float = 0

        for count in word_counts.values():
            total_words += count
            pow_total_words += pow(count, word_prob_power)

        return total_words, pow_total_words

    def _calculate_word_probabilities(
        self, word_counts: Dict[str, int], word_prob_power: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        word_probs: Dict[str, float] = {}
        pow_word_probs: Dict[str, float] = {}

        for word, count in word_counts.items():
            word_prob = count / self._total_words
            word_probs[word] = word_prob
            pow_word_prob = pow(word_prob, word_prob_power)
            pow_word_probs[word] = pow_word_prob

        return word_probs, pow_word_probs

    def should_keep_word(self, word: str) -> bool:
        if word not in self._word_probs:
            return False

        word_prob = self._word_probs[word]
        keep_prob: float = (np.sqrt(word_prob / self.sample_rate) + 1) * (
            self.sample_rate / word_prob
        )

        return np.random.random() <= keep_prob

    def sample_negatives(self, ignore_word: str, n_samples: int = 10) -> List[str]:
        # TODO Use word probabilities as a list.
        word_indices = np.choice(
            len(self._pow_word_probs), size=n_samples, p=self._pow_word_probs
        )
