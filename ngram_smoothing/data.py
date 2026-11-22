"""
Data preparation
================
Loads the Movie Review corpus and turns it into clean, tokenized sentences:

  - splits reviews into sentences and words (NLTK's pre-tokenized corpus)
  - lowercases all text
  - strips punctuation / special characters
  - drops near-empty sentences left over after cleaning

Sentence boundary markers (<s> ... </s>) are added later, inside the model,
because the number of start markers depends on the n-gram order.

About the dataset
-----------------
The "Movie Review Dataset" commonly found on Kaggle mirrors the classic
Cornell *Movie Review Polarity* corpus (Pang & Lee, 2004). NLTK ships that
same corpus as ``nltk.corpus.movie_reviews`` (2,000 reviews, ~1.58M tokens),
so it is used here directly. It downloads automatically on first use and
needs no Kaggle account or API key.
"""

import random
import string

RANDOM_SEED = 42

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NLTK_PACKAGES = {
    "movie_reviews": "corpora/movie_reviews",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}


def ensure_nltk_data():
    """Download the NLTK resources this project needs, if they are missing."""
    import nltk

    for package, resource_path in _NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


def clean_token(tok: str) -> str:
    """Lowercase a token and strip punctuation/special characters from it."""
    return tok.lower().translate(_PUNCT_TABLE)


def load_and_preprocess_sentences(n_reviews: int = None, min_len: int = 3):
    """
    Load the Movie Review corpus and return cleaned, tokenized sentences.

    Parameters
    ----------
    n_reviews : int or None
        If given, use a reproducible random sample of this many reviews
        (handy for quick experiments). If None, use all 2,000 reviews.
    min_len : int
        Drop sentences with fewer than this many tokens after cleaning
        (removes sentences that were nothing but punctuation).

    Returns
    -------
    List[List[str]]
        One list of lowercase, punctuation-free tokens per sentence.
    """
    ensure_nltk_data()
    from nltk.corpus import movie_reviews

    fileids = movie_reviews.fileids()
    if n_reviews is not None:
        rng = random.Random(RANDOM_SEED)
        fileids = rng.sample(fileids, n_reviews)

    sentences = []
    for fid in fileids:
        for raw_sent in movie_reviews.sents(fid):
            cleaned = [clean_token(tok) for tok in raw_sent]
            cleaned = [tok for tok in cleaned if tok]  # drop empty strings
            if len(cleaned) >= min_len:
                sentences.append(cleaned)
    return sentences


def train_test_split(sentences, test_ratio=0.2, seed=RANDOM_SEED):
    """Shuffle and split sentences into (train, test) lists."""
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test


def k_fold_split(sentences, k=5, seed=RANDOM_SEED):
    """Yield ``(train_sentences, test_sentences)`` for each of k folds."""
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    fold_size = len(shuffled) // k
    folds = [shuffled[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    # any remainder sentences go into the last fold
    folds[-1].extend(shuffled[k * fold_size:])

    for i in range(k):
        test = folds[i]
        train = [s for j, f in enumerate(folds) if j != i for s in f]
        yield train, test


if __name__ == "__main__":
    sents = load_and_preprocess_sentences(n_reviews=50)
    print(f"Loaded {len(sents)} sentences from 50 reviews")
    print("Example sentence:", sents[0])
    print("Example sentence:", sents[1])
