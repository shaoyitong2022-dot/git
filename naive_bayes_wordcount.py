"""
=========================================================================
  Naive Bayes Document Classification Prep: Binomial -> Poisson
=========================================================================
  In text classification (e.g., Naive Bayes for spam detection), we often
  model the count of a specific 'sensitive' word across documents.
  If each of the n words in a document independently has probability p
  of being that word, the count X ~ Binomial(n, p).

  When n is large and p is small, the Binomial distribution is well
  approximated by a Poisson distribution with rate lambda = n * p.

  This script simulates the process and visually compares the empirical
  histogram against the theoretical Poisson PMF for two values of n.
=========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from math import factorial


def simulate_word_counts(n: int, p: float = 0.03, trials: int = 10000) -> np.ndarray:
    samples = np.random.rand(trials, n) < p
    return samples.sum(axis=1)


def plot_comparison(n: int, p: float = 0.03, trials: int = 10000):
    data = simulate_word_counts(n, p, trials)
    lam = n * p

    plt.figure(figsize=(10, 5))

    max_count = data.max()
    bins = np.arange(0, max_count + 2) - 0.5
    plt.hist(data, bins=bins, density=True, alpha=0.65,
             color="steelblue", edgecolor="white",
             label=f"Simulation (trials={trials})")

    k = np.arange(0, max_count + 1)
    poisson_pmf = np.array([np.exp(-lam) * lam**int(x) / factorial(int(x)) for x in k])

    plt.plot(k, poisson_pmf, "r-", linewidth=2, label=f"Poisson(lambda={lam:.1f})")

    plt.xlabel("Sensitive word count")
    plt.ylabel("Probability density")
    plt.title(f"Binomial(n={n}, p={p}) vs Poisson(lambda={lam:.1f})", fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    print("Running: n = 1000 ...")
    plot_comparison(n=1000, p=0.03)

    print("Running: n = 10000 ...")
    plot_comparison(n=10000, p=0.03)


# =========================================================================
#  ANALYSIS AND ANSWERS
# =========================================================================
#
# Q: When n=1000 vs n=10000, how does the Binomial-to-Poisson fit change?
#    Why?
#
# A:
#   1) What we see in the plots:
#      - n = 1000:  The histogram roughly matches Poisson, but there are
#                   noticeable local deviations, especially in the tail
#                   regions (higher counts). The fit is "good" but not
#                   perfect.
#      - n = 10000: The histogram and the Poisson curve are nearly
#                   indistinguishable. The fit is excellent.
#
#   2) Why does this happen?
#
#      a) Poisson Limit Theorem (Law of Rare Events):
#         Let X ~ Binomial(n, p). As n -> infinity and p -> 0 while
#         keeping lambda = n*p constant, for any fixed k:
#              P(X = k) -> (lambda^k / k!) * exp(-lambda)
#         The Binomial converges pointwise to a Poisson distribution.
#
#      b) Convergence quality depends on n (and how small p is):
#         - The total variation distance between Binomial(n,p) and
#           Poisson(np) is bounded approximately by p:
#              d_TV( Bin(n,p), Pois(np) ) <= p
#           When p is fixed, this bound does not shrink with n alone.
#           However, the *pointwise* approximation quality does improve
#           as n grows larger because the higher-order terms in the
#           error expansion shrink as O(1/n).
#         - In our experiment p = 0.03 (not extremely small). With
#           n = 1000 (lambda = 30), the Binomial variance np(1-p) = 29.1
#           differs slightly from the Poisson variance 30. With
#           n = 10000 (lambda = 300), the relative difference between
#           variances becomes negligible.
#
#      c) Practical rule of thumb:
#         The Poisson approximation to the Binomial is considered
#         acceptable when n >= 20 and p <= 0.05, or when n >= 100 and
#         np <= 10. Both n=1000 and n=10000 satisfy this, but n=10000
#         gives a much tighter fit.
#
#      d) Edge effects for moderate p:
#         Since p = 0.03 is not truly "rare" (typical threshold < 0.01),
#         we need a larger n to compensate. n = 10000 provides enough
#         samples for the law of large numbers to smooth out the
#         Binomial discreteness and match Poisson shape.
#
#   3) Conclusion:
#      As n increases, the empirical Binomial distribution converges
#      closer to Poisson(np). This is a direct illustration of the
#      Poisson limit theorem: the Naive Bayes multinomial model with
#      rare-feature assumptions becomes increasingly well-approximated
#      by Poisson-based likelihoods when document lengths are large.
# =========================================================================
