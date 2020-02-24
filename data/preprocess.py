import os
from typing import List
import re
import string
import itertools

from pandarallel import pandarallel
import pandas as pd
import numpy as np
import spacy

pandarallel.initialize(progress_bar=True)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

NLP = spacy.load("en")
SENTENCE_LENGTH_THRESHOLD = 4


def extract_sentence(row: pd.Series) -> List[str]:
    sentences = []
    for sentence in NLP(row.review).sents:
        clean_sentence = re.sub("[^a-zA-Z\s]", "", sentence.string).strip().lower()
        sentence_length = len(clean_sentence.split())
        if sentence_length >= SENTENCE_LENGTH_THRESHOLD:
            sentences.append(clean_sentence)
    return sentences


if __name__ == "__main__":
    dataset_path = os.path.join(DATA_DIR, "raw", "amazon_reviews.csv")

    df = pd.read_csv(dataset_path, header=None, usecols=[2], names=["review"])

    sentences_series = df.parallel_apply(extract_sentence, axis=1)
    sentences = list(
        itertools.chain.from_iterable((sentence for sentence in sentences_series))
    )

    sents_df = pd.DataFrame(sentences)

    out_path = os.path.join(DATA_DIR, "processed", "amazon_reviews.csv")
    sents_df.to_csv(out_path, index=False, header=None)

    word_stats = sents_df[0].str.split().map(lambda sent: len(sent)).describe()

    print("\nSentence word stats:\n")
    print(word_stats)
