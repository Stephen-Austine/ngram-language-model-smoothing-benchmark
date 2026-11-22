"""Fast unit tests on a tiny hand-written corpus (no NLTK download needed)."""

import math

import pytest

from ngram_smoothing import (
    END,
    START,
    NGramLanguageModel,
    clean_token,
    k_fold_split,
    train_test_split,
)
from ngram_smoothing.model import get_ngrams, pad_sentence

CORPUS = [
    "the movie was good".split(),
    "the movie was bad".split(),
    "the film was very good".split(),
    "this film was good and the plot was good".split(),
    "the plot was thin but the acting was good".split(),
    "a very good movie with a good plot".split(),
    "the acting was bad and the plot was thin".split(),
    "this movie was very bad".split(),
    "the film was good but the ending was bad".split(),
    "a good film with a very good cast".split(),
    "the cast was good and the movie was fun".split(),
    "the ending was thin but the movie was good".split(),
]  # every sentence is distinct, so singleton n-grams exist (Good-Turing needs them)


# ---------------------------------------------------------------- helpers
def test_clean_token_lowercases_and_strips_punctuation():
    assert clean_token("Hello,") == "hello"
    assert clean_token("--") == ""
    assert clean_token("don't") == "dont"


def test_pad_sentence_uses_n_minus_one_starts_and_one_end():
    assert pad_sentence(["a", "b"], 3) == [START, START, "a", "b", END]
    assert pad_sentence(["a", "b"], 1) == [START, "a", "b", END]  # at least one <s>


def test_get_ngrams():
    assert list(get_ngrams(["a", "b", "c"], 2)) == [("a", "b"), ("b", "c")]


def test_train_test_split_is_deterministic_and_partitions():
    sents = [[str(i), "x", "y"] for i in range(100)]
    train1, test1 = train_test_split(sents, 0.2)
    train2, test2 = train_test_split(sents, 0.2)
    assert (train1, test1) == (train2, test2)
    assert len(test1) == 20 and len(train1) == 80
    assert sorted(train1 + test1) == sorted(sents)


def test_k_fold_split_covers_every_sentence_once_as_test():
    sents = [[str(i), "x", "y"] for i in range(103)]  # remainder goes to last fold
    seen = []
    for train, test in k_fold_split(sents, k=5):
        assert not set(map(tuple, train)) & set(map(tuple, test))
        seen.extend(map(tuple, test))
    assert sorted(seen) == sorted(map(tuple, sents))


# ------------------------------------------------------------ probabilities
@pytest.mark.parametrize("method", ["laplace", "kneser_ney"])
def test_bigram_distribution_sums_to_one(method):
    """P(. | 'the') must be a proper distribution over the vocabulary."""
    lm = NGramLanguageModel(2).fit(CORPUS)
    total = sum(lm.prob(("the", w), method) for w in lm.vocab)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_kneser_ney_trigram_distribution_is_normalised_up_to_a_tiny_leak():
    """
    For n >= 3 the unigram-continuation base distribution counts the <s> marker
    as a continuation but excludes it from V, so the distribution over V is off
    by ~1/(#bigram types + V). Negligible on the real corpus (~1e-6) but visible
    on this 12-sentence toy corpus, hence the looser tolerance.
    """
    lm = NGramLanguageModel(3).fit(CORPUS)
    total = sum(lm.prob(("the", "movie", w), "kneser_ney") for w in lm.vocab)
    assert total == pytest.approx(1.0, abs=2e-2)
    assert total <= 1.0 + 1e-9


@pytest.mark.parametrize("method", ["laplace", "good_turing", "kneser_ney"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_probabilities_are_valid(n, method):
    lm = NGramLanguageModel(n).fit(CORPUS)
    ngram = tuple(["the"] * (n - 1) + ["movie"])
    p = lm.prob(ngram, method)
    assert 0.0 < p <= 1.0


@pytest.mark.parametrize("method", ["laplace", "good_turing", "kneser_ney"])
def test_unseen_ngram_still_gets_nonzero_probability(method):
    lm = NGramLanguageModel(2).fit(CORPUS)
    assert lm.prob(("plot", "cast"), method) > 0.0
    assert lm.prob(("zzz", "qqq"), method) > 0.0  # completely out of vocabulary


def test_kneser_ney_prefers_seen_over_unseen_bigram():
    lm = NGramLanguageModel(2).fit(CORPUS)
    assert lm.prob(("the", "movie"), "kneser_ney") > lm.prob(("the", "acting"), "kneser_ney")


def test_custom_kneser_ney_discount_is_used():
    lm = NGramLanguageModel(2, kn_discount=0.5).fit(CORPUS)
    assert all(d == 0.5 for d in lm._kn_discounts.values())


# --------------------------------------------------------------- perplexity
@pytest.mark.parametrize("method", ["laplace", "good_turing", "kneser_ney"])
def test_perplexity_is_lower_on_seen_text_than_on_gibberish(method):
    lm = NGramLanguageModel(2).fit(CORPUS)
    seen = lm.perplexity(CORPUS[:4], method)
    gibberish = lm.perplexity([["qqq", "zzz", "xxx", "www"]], method)
    assert math.isfinite(seen) and math.isfinite(gibberish)
    assert seen < gibberish


def test_perplexity_of_empty_test_set_is_infinite():
    lm = NGramLanguageModel(2).fit(CORPUS)
    assert lm.perplexity([], "laplace") == float("inf")


def test_unknown_method_raises():
    lm = NGramLanguageModel(2).fit(CORPUS)
    with pytest.raises(ValueError):
        lm.prob(("the", "movie"), "witten_bell")


# --------------------------------------------------------------- generation
def test_generate_is_reproducible_with_a_seed():
    lm = NGramLanguageModel(3).fit(CORPUS)
    assert lm.generate(max_tokens=10, seed=7) == lm.generate(max_tokens=10, seed=7)
    assert isinstance(lm.generate(max_tokens=10, seed=7), str)


def test_good_turing_reserves_no_unseen_mass_when_there_are_no_singletons():
    """Good-Turing's unseen mass is N1/N, so a corpus without singletons reserves none."""
    no_singletons = [["a", "b", "c"]] * 5
    lm = NGramLanguageModel(2).fit(no_singletons)
    assert lm._gt_unseen_mass[2] == 0.0
    assert lm.prob(("a", "c"), "good_turing") == 0.0  # documented limitation of the naive variant
