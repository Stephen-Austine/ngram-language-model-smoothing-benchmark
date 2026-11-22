"""
Diagnostics behind the numbers quoted in the README's "Analysis" and "Limitations" sections.

    python scripts/diagnostics.py | tee results/diagnostics.txt

Reports, on the same 80/20 split used by the main experiment:
  1. how many test n-grams were never seen in training (per order)
  2. how often the 1e-12 probability floor is hit, and perplexity with / without it
  3. how much probability each distribution gives the 10 most frequent tokens
  4. how many events a unigram model scores vs. higher orders (the <s> marker)
  5. the out-of-vocabulary rate of the test split
  6. the size of the Kneser-Ney normalisation quirk for n >= 3
"""

import math

from ngram_smoothing import NGramLanguageModel, load_and_preprocess_sentences, train_test_split
from ngram_smoothing.model import get_ngrams, pad_sentence

METHODS = ("laplace", "good_turing", "kneser_ney")


def main():
    sentences = load_and_preprocess_sentences()
    train, test = train_test_split(sentences, 0.2)
    print(f"train={len(train):,} sentences | test={len(test):,} sentences\n")

    print("1-2. Unseen test n-grams and probability-floor usage")
    for n in (2, 3, 4):
        lm = NGramLanguageModel(n).fit(train)
        for method in METHODS:
            total = floored = unseen = 0
            bits_floor = bits_nofloor = 0.0
            for s in test:
                for ng in get_ngrams(pad_sentence(s, n), n):
                    p = lm.prob(ng, method)
                    total += 1
                    unseen += lm.counts[n].get(ng, 0) == 0
                    floored += p < 1e-12
                    bits_floor += -math.log2(max(p, 1e-12))
                    bits_nofloor += -math.log2(p) if p > 0 else math.inf
            pp_nofloor = 2 ** (bits_nofloor / total) if bits_nofloor != math.inf else math.inf
            print(f"  n={n} {method:<12} unseen {unseen / total:6.1%} | floored {floored / total:6.1%} | "
                  f"PP with floor {2 ** (bits_floor / total):>10,.0f} | without {pp_nofloor:>10,.0f}")

    print("\n3. Probability mass on the 10 most frequent tokens (unigram level)")
    lm1 = NGramLanguageModel(1).fit(train)
    top = [w for w, _ in lm1.counts[1].most_common(12) if w[0] != "<s>"][:10]
    print("  tokens:", [w[0] for w in top])
    print(f"  raw frequency   {sum(lm1.counts[1][w] for w in top) / lm1.total_unigrams:.3f}")
    print(f"  Laplace         {sum(lm1.laplace_prob(w) for w in top):.3f}")
    print(f"  KN continuation {sum(lm1.kneser_ney_prob(w) for w in top):.3f}")

    print("\n4. Events scored per test set")
    e1 = sum(len(list(get_ngrams(pad_sentence(s, 1), 1))) for s in test)
    e2 = sum(len(list(get_ngrams(pad_sentence(s, 2), 2))) for s in test)
    print(f"  n=1: {e1:,}  n>=2: {e2:,}  (unigram scores {e1 - e2:,} extra <s> events, {(e1 - e2) / e1:.1%})")

    print("\n5. Out-of-vocabulary rate")
    oov = sum(1 for s in test for w in s if w not in lm1.vocab)
    tokens = sum(len(s) for s in test)
    print(f"  {oov:,} / {tokens:,} test tokens = {oov / tokens:.2%}")

    print("\n6. Kneser-Ney normalisation quirk (n >= 3)")
    lm4 = NGramLanguageModel(4).fit(train)
    types, v = lm4._kn_total_bigram_types, len(lm4.vocab)
    print(f"  bigram types={types:,}  V={v:,}  1/(types+V) = {1 / (types + v):.2e}")


if __name__ == "__main__":
    main()
