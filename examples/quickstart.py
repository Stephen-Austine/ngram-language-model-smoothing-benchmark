"""
Quickstart: fit one model, inspect probabilities, compare smoothers, generate text.

    python examples/quickstart.py
"""

from ngram_smoothing import NGramLanguageModel, load_and_preprocess_sentences, train_test_split

# 1. Load a reproducible 300-review sample (use n_reviews=None for all 2,000)
sentences = load_and_preprocess_sentences(n_reviews=300)
train, test = train_test_split(sentences, test_ratio=0.2)
print(f"{len(train):,} training sentences | {len(test):,} test sentences")

# 2. Fit a trigram model (counts for orders 1, 2 and 3 are collected in one pass)
lm = NGramLanguageModel(3).fit(train)
print(f"vocabulary size: {len(lm.vocab):,}\n")

# 3. P(word | two previous words) under each smoothing technique
ngram = ("this", "is", "a")
print(f"P(w | 'this is') for the trigram {ngram}")
for method in ("laplace", "good_turing", "kneser_ney"):
    print(f"  {method:<12} {lm.prob(ngram, method):.6f}")

# 4. Held-out perplexity (lower is better)
print("\nTrigram perplexity on held-out sentences")
for method in ("laplace", "good_turing", "kneser_ney"):
    print(f"  {method:<12} {lm.perplexity(test, method):>12,.1f}")

# 5. Sample text from the raw trigram distribution
print("\nSampled sentences")
for seed in (1, 2, 3):
    print("  ", lm.generate(max_tokens=15, seed=seed))
