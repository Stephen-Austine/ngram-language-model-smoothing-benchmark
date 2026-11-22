"""
N-gram language model with three smoothing techniques
=====================================================
Implements an n-gram language model from raw counts, three smoothing
strategies (Laplace / Add-One, Good-Turing, and interpolated Kneser-Ney),
and perplexity-based evaluation.

The model always builds counts for EVERY order from 1..n (not just order n)
because:
  - Good-Turing here re-estimates counts order-by-order and derives a
    conditional probability from the ratio of two smoothed joint
    probabilities (order n over order n-1).
  - Kneser-Ney is defined recursively: the order-k estimate backs off to
    the order-(k-1) continuation estimate, all the way down to unigrams.
"""

import math
from collections import Counter, defaultdict

START = "<s>"
END = "</s>"


def pad_sentence(sentence, n):
    """Pad a sentence with (n-1) start tokens and one end token."""
    pad_n = max(n - 1, 1)
    return [START] * pad_n + sentence + [END]


def get_ngrams(tokens, n):
    """Yield n-grams (as tuples) from a padded list of tokens."""
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


class NGramLanguageModel:
    """
    An n-gram language model that supports Laplace, Good-Turing, and
    Kneser-Ney smoothing on top of the same underlying counts.
    """

    def __init__(self, n, kn_discount=None):
        self.n = n
        self.kn_discount = kn_discount  # if None, estimated from data per order
        # counts[k] : Counter of k-length tuples -> frequency, for k = 1..n
        # NOTE: we always keep at least order-2 (bigram) counts around, even
        # for a unigram (n=1) model, because Kneser-Ney's continuation trick
        # is defined in terms of bigram statistics (how many distinct
        # contexts a word appears in). Without this, a "unigram KN model"
        # would have nothing to base continuation counts on.
        self._count_max_order = max(n, 2)
        self.counts = {k: Counter() for k in range(1, self._count_max_order + 1)}
        self.vocab = set()
        self.total_unigrams = 0

        # Good-Turing support structures, filled in by _fit_good_turing()
        self._gt_star = {k: {} for k in range(1, n + 1)}       # count -> c*
        self._gt_unseen_mass = {k: 0.0 for k in range(1, n + 1)}
        self._gt_norm = {k: 0.0 for k in range(1, n + 1)}      # sum of c* (seen)
        self._gt_num_possible = {k: 0 for k in range(1, n + 1)}
        self._gt_total_count = {k: 0 for k in range(1, n + 1)}  # cached sum(counts[k].values())

        # Kneser-Ney support structures
        self._kn_discounts = {}                      # order -> discount D
        self._kn_continuation_counts = Counter()      # word -> N1+(*w)   (unigram level)
        self._kn_total_bigram_types = 0                # N1+(**)
        # for order k>=2: number of distinct words following each (k-1)-context
        self._kn_context_type_counts = {k: defaultdict(int) for k in range(2, n + 1)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, sentences):
        """Count n-grams of every order 1..n from the training sentences."""
        for sentence in sentences:
            padded = pad_sentence(sentence, self._count_max_order)
            for k in range(1, self._count_max_order + 1):
                for ng in get_ngrams(padded, k):
                    self.counts[k][ng] += 1
            for tok in sentence:
                self.vocab.add(tok)
        self.vocab.add(END)
        self.total_unigrams = sum(self.counts[1].values())

        self._fit_good_turing()
        self._fit_kneser_ney()
        return self

    # ------------------------------------------------------------------
    # Laplace (Add-One) smoothing
    # ------------------------------------------------------------------
    def laplace_prob(self, ngram):
        """P(w_n | w_1..w_n-1) with add-one smoothing."""
        n = len(ngram)
        V = len(self.vocab)
        count_ngram = self.counts[n].get(ngram, 0)
        if n == 1:
            count_context = self.total_unigrams
        else:
            context = ngram[:-1]
            count_context = self.counts[n - 1].get(context, 0)
        return (count_ngram + 1) / (count_context + V)

    # ------------------------------------------------------------------
    # Good-Turing discounting
    # ------------------------------------------------------------------
    def _fit_good_turing(self):
        """
        For each order k, re-estimate counts using the Good-Turing formula:
            c* = (c + 1) * N_{c+1} / N_c
        where N_c is the number of distinct k-grams occurring exactly c
        times. Falls back to the raw count c when N_{c+1} is unavailable
        (a standard practical fix for the sparse high-count tail).
        The probability mass freed up (N_1 / N_total) is reserved for
        unseen k-grams and spread uniformly across the estimated number of
        unseen k-gram types.
        """
        V = len(self.vocab)
        for k in range(1, self.n + 1):
            freqs = self.counts[k]
            if not freqs:
                continue
            N_total = sum(freqs.values())
            self._gt_total_count[k] = N_total
            # frequency-of-frequencies: Nc[c] = number of k-gram TYPES seen exactly c times
            Nc = Counter(freqs.values())
            max_c = max(Nc.keys())

            star = {}
            for c in range(1, max_c + 1):
                if Nc.get(c, 0) == 0:
                    continue
                if Nc.get(c + 1, 0) > 0:
                    star[c] = (c + 1) * Nc[c + 1] / Nc[c]
                else:
                    star[c] = c  # fallback: no data to discount further
            self._gt_star[k] = star

            # total re-estimated mass: sum of c* over every observed k-gram type
            seen_mass = sum(star[c] for c in freqs.values())
            self._gt_norm[k] = seen_mass

            n1 = Nc.get(1, 0)
            self._gt_unseen_mass[k] = n1 / N_total if N_total > 0 else 0.0

            # estimated size of the full event space of order-k n-grams = V^k
            possible = V ** k
            unseen_types = max(possible - len(freqs), 1)
            self._gt_num_possible[k] = unseen_types

    def _gt_joint_prob(self, ngram):
        """Good-Turing-smoothed JOINT probability P(ngram) for its order."""
        k = len(ngram)
        freqs = self.counts[k]
        c = freqs.get(ngram, 0)
        N_total = self._gt_total_count.get(k, 1) or 1
        if c > 0:
            c_star = self._gt_star[k].get(c, c)
            return c_star / N_total
        else:
            # split the reserved "unseen" mass uniformly over unseen types
            unseen_mass = self._gt_unseen_mass[k]
            unseen_types = self._gt_num_possible[k]
            return unseen_mass / unseen_types

    def good_turing_prob(self, ngram):
        """P(w_n | w_1..w_n-1) derived from ratio of Good-Turing joint probs."""
        n = len(ngram)
        if n == 1:
            return self._gt_joint_prob(ngram)
        context = ngram[:-1]
        p_joint = self._gt_joint_prob(ngram)
        p_context = self._gt_joint_prob(context)
        if p_context <= 0:
            return p_joint
        prob = p_joint / p_context
        # p_context here is itself a probability (not raw count based), the
        # ratio can occasionally exceed 1 for very sparse contexts; clip to
        # keep it a valid-looking probability.
        return min(prob, 1.0)

    # ------------------------------------------------------------------
    # Kneser-Ney (interpolated) smoothing
    # ------------------------------------------------------------------
    def _fit_kneser_ney(self):
        """
        Precompute the continuation-count statistics needed for
        interpolated Kneser-Ney at every order, and estimate a discount D
        per order using the Chen & Goodman (1999) approximation:
            D = n1 / (n1 + 2*n2)
        where n1, n2 are the number of n-gram types with count exactly 1
        and 2 respectively.
        """
        # discount per order
        for k in range(1, self.n + 1):
            freqs = self.counts[k]
            n1 = sum(1 for c in freqs.values() if c == 1)
            n2 = sum(1 for c in freqs.values() if c == 2)
            if n1 + 2 * n2 == 0:
                self._kn_discounts[k] = 0.75  # safe default
            else:
                D = n1 / (n1 + 2 * n2)
                self._kn_discounts[k] = min(max(D, 0.1), 0.9)  # keep in sane range

        if self.kn_discount is not None:
            for k in self._kn_discounts:
                self._kn_discounts[k] = self.kn_discount

        # unigram continuation counts: N1+(* w) = number of distinct single
        # preceding words seen before w (uses bigram counts)
        bigrams = self.counts.get(2, Counter())
        for (w_prev, w), c in bigrams.items():
            self._kn_continuation_counts[w] += 1
        self._kn_total_bigram_types = len(bigrams)  # N1+(**)

        # for k >= 2: number of distinct continuations of each (k-1)-context
        for k in range(2, self.n + 1):
            for ng in self.counts[k]:
                context = ng[:-1]
                self._kn_context_type_counts[k][context] += 1

    def _kn_continuation_prob(self, word):
        """
        Lowest-order Kneser-Ney estimate: continuation probability, i.e.
        how many distinct contexts a word appears after, rather than its
        raw frequency. Add-one smoothed over the space of bigram types so
        that a word never seen even as a continuation (fully OOV) still
        gets a small, non-degenerate probability instead of ~0.
        """
        V = len(self.vocab) or 1
        num = self._kn_continuation_counts.get(word, 0)
        return (num + 1) / (self._kn_total_bigram_types + V)

    def kneser_ney_prob(self, ngram):
        """
        Interpolated Kneser-Ney: P(w | context) recursively backs off to
        lower orders, discounting seen counts by D and redistributing the
        freed mass, weighted by how many distinct words followed context.
        """
        return self._kn_prob_recursive(ngram)

    def _kn_prob_recursive(self, ngram):
        k = len(ngram)
        if k == 1:
            return self._kn_continuation_prob(ngram[0])

        context = ngram[:-1]
        context_count = self.counts[k - 1].get(context, 0)
        D = self._kn_discounts.get(k, 0.75)

        if context_count == 0:
            # never saw this context at all -> back off completely
            return self._kn_prob_recursive(ngram[1:])

        c_ngram = self.counts[k].get(ngram, 0)
        discounted = max(c_ngram - D, 0) / context_count

        num_continuations = self._kn_context_type_counts[k].get(context, 0)
        lam = (D * num_continuations) / context_count

        lower = self._kn_prob_recursive(ngram[1:])
        return discounted + lam * lower

    # ------------------------------------------------------------------
    # Probability dispatch + perplexity
    # ------------------------------------------------------------------
    def prob(self, ngram, method):
        if method == "laplace":
            return self.laplace_prob(ngram)
        elif method == "good_turing":
            return self.good_turing_prob(ngram)
        elif method == "kneser_ney":
            return self.kneser_ney_prob(ngram)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def sentence_log_prob(self, sentence, method):
        """Sum of log2 probabilities of every n-gram in a padded sentence."""
        padded = pad_sentence(sentence, self.n)
        log_prob = 0.0
        n_tokens = 0
        for ng in get_ngrams(padded, self.n):
            p = self.prob(ng, method)
            p = max(p, 1e-12)  # numerical floor, avoids log(0)
            log_prob += math.log2(p)
            n_tokens += 1
        return log_prob, n_tokens

    def perplexity(self, sentences, method):
        """
        Corpus-level perplexity:
            PP = 2 ** ( - (1/N) * sum(log2 P(w_i | context)) )
        computed over all n-grams in all test sentences.
        """
        total_log_prob = 0.0
        total_tokens = 0
        for sentence in sentences:
            lp, n_tok = self.sentence_log_prob(sentence, method)
            total_log_prob += lp
            total_tokens += n_tok
        if total_tokens == 0:
            return float("inf")
        avg_log_prob = total_log_prob / total_tokens
        return 2 ** (-avg_log_prob)

    # ------------------------------------------------------------------
    # Bonus: qualitative text generation (illustrates n's effect on fluency)
    # ------------------------------------------------------------------
    def generate(self, max_tokens=20, seed=None):
        """
        Randomly sample a sentence from the raw (unsmoothed) order-n
        distribution, weighted by observed counts. Useful only as a
        qualitative illustration of how n-gram size affects fluency /
        coherence -- not part of the formal perplexity evaluation.
        """
        import random
        rng = random.Random(seed)
        prefix_len = self.n - 1
        context = [START] * max(prefix_len, 1)
        output = []
        top_level = self.counts[self.n]

        for _ in range(max_tokens):
            ctx_tuple = tuple(context[-prefix_len:]) if prefix_len > 0 else tuple()
            candidates = {ng[-1]: c for ng, c in top_level.items() if ng[:prefix_len] == ctx_tuple}
            if not candidates:
                break
            words, weights = zip(*candidates.items())
            next_word = rng.choices(words, weights=weights, k=1)[0]
            if next_word == END:
                break
            output.append(next_word)
            context.append(next_word)
        return " ".join(output)


if __name__ == "__main__":
    # quick smoke test
    from ngram_smoothing.data import load_and_preprocess_sentences, train_test_split

    sents = load_and_preprocess_sentences(n_reviews=20)
    train, test = train_test_split(sents, test_ratio=0.2)

    for n in [1, 2, 3]:
        lm = NGramLanguageModel(n).fit(train)
        for method in ["laplace", "good_turing", "kneser_ney"]:
            pp = lm.perplexity(test[:50], method)
            print(f"n={n:>1} | {method:<12} | perplexity = {pp:8.2f}")
