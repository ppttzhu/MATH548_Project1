# _________________LIBRARIES_________________
import numpy as np
import matplotlib.pyplot as plt  # for plotting graphs
import math
from scipy.stats import norm


# ________________FUNCTIONS______________________

# Binomial tree

def binomial_tree_analytic(cp: int, s: float, k: float, t: float, r: float, sigma: float, n: int) -> float:
    """
    Binomial tree method to price call and put option
    :param cp: indicator 1 for call, -1 for put
    :param s: spot price of the underlying asset
    :param k: strike price
    :param t: time to maturity (expressed in years)
    :param r: risk free rate (annual rate, expressed in terms of continuous compounding)
    :param sigma: volatility of returns of the underlying asset
    :param n: height of the binomial tree
    :return: the price of call or put option
    """

    delta_t = t / n

    up = math.exp(sigma * math.sqrt(delta_t))
    down = math.exp(-sigma * math.sqrt(delta_t))

    # risk neutral probability
    q_up = (math.exp(r * delta_t) - down) / (up - down)
    q_down = (up - math.exp(r * delta_t)) / (up - down)

    npv = 0
    # price by adding discounted payoff
    for i in range(n + 1):
        payoff = 0
        if cp == 1:
            payoff = call_payoff((s * math.pow(up, i) * math.pow(down, n - i)), k)
        elif cp == -1:
            payoff = put_payoff((s * math.pow(up, i) * math.pow(down, n - i)), k)
        npv += math.exp(-r * t) * yang_hui_triangle(i, n) * payoff * math.pow(q_up, i) * math.pow(q_down, n - i)

    return npv


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

def bs_formula(cp: int, s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Black–Scholes formula to price call and put option
    https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
    :param cp: indicator 1 for call, -1 for put
    :param s: spot price of the underlying asset
    :param k: strike price
    :param t: time to maturity (expressed in years)
    :param r: risk free rate (annual rate, expressed in terms of continuous compounding)
    :param sigma: volatility of returns of the underlying asset
    :return: the price of call or put option
    """
    d1 = bs_formula_d1(s, k, t, r, sigma)
    d2 = bs_formula_d2(s, k, t, r, sigma)

    npv = cp * (norm.cdf(cp * d1) * s - norm.cdf(cp * d2) * k * math.exp(-r * t))

    return npv


# Supporting functions

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


def call_payoff(s: float, k: float) -> float:
    """
    Calculate the payoff of call option
    :param s: spot price
    :param k: strike price
    :return: payoff of call option
    """
    payoff = 0
    if s > k:
        payoff = s - k

    return payoff


def put_payoff(s: float, k: float) -> float:
    """
    Calculate the payoff of put option
    :param s: spot price
    :param k: strike price
    :return: payoff of put option
    """
    payoff = 0
    if s < k:
        payoff = k - s

    return payoff


# ------------Data Retrieval & Prep------------

# ------------Inputs & Outputs------------

# ------------Print Solution------------

# -------------Graph Solution------------

# -------------Test------------
hh1 = binomial_tree_analytic(-1, 10, 10, 1, 0.01, 0.1, 10)
hh2 = bs_formula(-1, 10, 10, 1, 0.01, 0.1)
print(hh1)
print(hh2)
