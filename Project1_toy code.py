# _________________LIBRARIES_________________
import numpy as np
import matplotlib.pyplot as plt  # for plotting graphs
import math
from scipy.stats import norm


# ________________FUNCTIONS______________________

# Binomial tree

def binomial_tree_call_analytic(s: float, k: float, t: float, r: float, sigma: float, n: int) -> float:
    """
    Binomial tree method to price call option
    :param s: spot price of the underlying asset
    :param k: strike price
    :param t: time to maturity (expressed in years)
    :param r: risk free rate (annual rate, expressed in terms of continuous compounding)
    :param sigma: volatility of returns of the underlying asset
    :param n: height of the binomial tree
    :return: the price of call option
    """

    delta_t = t / n

    up = math.exp(sigma * math.sqrt(delta_t))
    down = math.exp(-sigma * math.sqrt(delta_t))

    q_up = (math.exp(r * delta_t) - down) / (up - down)
    q_down = (up - math.exp(r * delta_t)) / (up - down)

    call = 0
    for i in range(n + 1):
        payoff = 0
        if (s * math.pow(up, i) * math.pow(down, n - i)) > k:
            payoff = (s * math.pow(up, i) * math.pow(down, n - i)) - k
        call += math.exp(-r * t) * yang_hui_triangle(i, n) * payoff * math.pow(q_up, i) * math.pow(q_down, n - i)

    return call


def binomial_tree_hedging(s: list, x: list, r: float, t: float) -> list:
    """
    private binomial tree support function to calculate hedging ratio on each node
    :param s: list of spot price of the underlying asset. Length 2, upward and downward side.
    :param x: list of value of the contingent claim. Length 2, upward and downward side.
    :param r: risk free rate (annual rate, expressed in terms of compounding)
    :param t: time interval of binomial tree (expressed in years)
    :return: list of hedging strategy of the contingent claim. Length 2, (H0, H1)
    """
    h1 = (x[1] - x[0]) / (s[1] - s[0])
    h0 = (x[1] - s[1] * h1) / math.pow((1 + r), t)

    hedging = [h0, h1]

    return hedging


# Black–Scholes formula to test in continuous time

def bs_formula_call(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Black–Scholes formula to price call option
    https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
    :param s: spot price of the underlying asset
    :param k: strike price
    :param t: time to maturity (expressed in years)
    :param r: risk free rate (annual rate, expressed in terms of continuous compounding)
    :param sigma: volatility of returns of the underlying asset
    :return: the price of call option
    """
    d1 = bs_formula_d1(s, k, t, r, sigma)
    d2 = bs_formula_d2(s, k, t, r, sigma)

    call = norm.cdf(d1) * s - norm.cdf(d2) * K * math.exp(-r * t)

    return call


def bs_formula_put(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Black–Scholes formula to price put option
    https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
    :param s: spot price of the underlying asset
    :param k: strike price
    :param t: time to maturity (expressed in years)
    :param r: risk free rate (annual rate, expressed in terms of continuous compounding)
    :param sigma: volatility of returns of the underlying asset
    :return: the price of put option
    """
    d1 = bs_formula_d1(s, k, t, r, sigma)
    d2 = bs_formula_d2(s, k, t, r, sigma)

    put = norm.cdf(-d2) * K * math.exp(-r * t) - norm.cdf(-d1) * s

    return put


def bs_formula_d1(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    private Black–Scholes formula support function
    """
    d1 = (math.log(s / k) + (r + sigma * sigma / 2) * t) / sigma * math.sqrt(t)

    return d1


def bs_formula_d2(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    private Black–Scholes formula support function
    """
    d2 = (math.log(s / k) + (r - sigma * sigma / 2) * t) / sigma * math.sqrt(t)

    return d2


# Supporting functions


def yang_hui_triangle(n: int, k: int) -> int:
    """
    Calculate C(n, k) using Yang Hui's (Pascal's) triangle
    https://en.wikipedia.org/wiki/Pascal%27s_triangle#Binomial_expansions
    :param n: select n
    :param k: total number k
    :return: the number of way to select n from k
    """

    # parameter out of range
    if n > k or k < 0 or n < 0:
        return 0

    list1 = []
    for i in range(k + 1):
        list0 = list1
        list1 = []
        for j in range(i + 1):
            if j == 0 or j == i:
                list1.append(1)
            else:
                list1.append(list0[j - 1] + list0[j])

    return list1[n]


def volatility_historical_price(s: list, multiplied_factor: int):
    """
    private support function to calculate volatility for underlying asset
    :param s: list of historical spot price of the underlying asset.
    :param multiplied_factor: number of business days for the corresponding frequency of historical data.
    E.g. for daily, 250 or 260. For weekly, 52.
    :return: annualized volatility
    """

    # insufficient length of data
    if len(s) < 2:
        return 0

    # logarithm of historical price
    log_s = []
    for i in range(1, len(s)):
        log_s.append(math.log(s[i - 1] / s[i]))

    vol = np.std(log_s, ddof=1) * math.sqrt(multiplied_factor)

    return vol


# ------------Data Retrieval & Prep------------

# ------------Inputs & Outputs------------

# ------------Print Solution------------

# -------------Graph Solution------------

# -------------Test------------
hh1 = binomial_tree_call_analytic(10, 10, 1, 0.01, 0.1, 10)
print(hh1)
