"""
Sample sentences from raw n-gram models of order 1-4 trained on the full corpus.

    python examples/generate_text.py

Sampling uses the raw (unsmoothed) next-word distribution, so this is a
qualitative illustration of how context length affects fluency -- it is not
part of the perplexity evaluation.
"""

from ngram_smoothing import NGramLanguageModel, load_and_preprocess_sentences, train_test_split

sentences = load_and_preprocess_sentences(n_reviews=None)
train, _ = train_test_split(sentences, test_ratio=0.2)

for n in (1, 2, 3, 4):
    lm = NGramLanguageModel(n).fit(train)
    print(f"n={n}:")
    for seed in (1, 2):
        print("  ", lm.generate(max_tokens=15, seed=seed))
    print()
