from typing import Dict, Tuple, List

import numpy as np


class Sampler:
    def __init__(
        self,
        word_counts: Dict[str, int],
        index_word: Dict[int, str],
        sample_rate: float = 1e-3,
        word_prob_power: float = 0.75,  # 3/4 as in the original word2vec paper.
    ):
        self._index_word: Dict[int, str] = index_word
        self._sample_rate: float = sample_rate

        total_words, pow_total_words = self._count_total_words(
            word_counts, word_prob_power
        )
        self._total_words: int = total_words
        self._pow_total_words: float = pow_total_words

        word_probs, pow_word_probs = self._calculate_word_probabilities(
            word_counts, word_prob_power
        )
        self._word_probs: Dict[str, float] = word_probs
        self._pow_word_probs: Dict[str, float] = pow_word_probs

        self._list_pow_word_probs: List[float] = [0] * len(index_word)
        for index, word in index_word.items():
            # index - 1 for OOV token that has 0 index.
            self._list_pow_word_probs[index - 1] = self._pow_word_probs[word]

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
            pow_word_prob = pow(count, word_prob_power) / self._pow_total_words
            pow_word_probs[word] = pow_word_prob

        return word_probs, pow_word_probs

    def should_keep_word(self, word: str) -> bool:
        if word not in self._word_probs:
            return False

        word_prob = self._word_probs[word]
        keep_prob: float = (np.sqrt(word_prob / self._sample_rate) + 1) * (
            self._sample_rate / word_prob
        )

        return np.random.random() <= keep_prob

    def sample_negatives(
        self, ignore_word: str = None, n_samples: int = 10, return_indices: bool = False
    ) -> List[str]:
        word_indices = np.random.choice(
            len(self._list_pow_word_probs), size=n_samples, p=self._list_pow_word_probs
        )

        if return_indices:
            return word_indices

        # index + 1 because if OOV token that has 0 index.
        negative_words = [self._index_word[index + 1] for index in word_indices]
        return negative_words
