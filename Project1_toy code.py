# _________________LIBRARIES_________________
import numpy as np
import matplotlib.pyplot as plt  # for plotting graphs
import math
from scipy.stats import norm


# ________________FUNCTIONS______________________

# Binominal tree

def binominal_tree_hedging(s: list, x: list, r: float, t: float) -> list:
    """
    private binominal tree support function to calculate hedging ratio on each node
    :param s: list of spot price of the underlying asset. Length 2, upward and downward side.
    :param x: list of value of the contingent claim. Length 2, upward and downward side.
    :param r: risk free rate (annual rate, expressed in terms of compounding)
    :param t: time interval of binominal tree (expressed in years)
    :return: list of hedging strategy of the contingent claim. Length 2, (H0, H1)
    """
    H1 = (x[1] - x[0]) / (s[1] - s[0])
    H0 = (x[1] - s[1] * H1) / math.pow((1 + r), t)

    hedging = [H0, H1]

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
    :return: the price of call option
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

# ------------Data Retrieval & Prep------------

# ------------Inputs & Outputs------------

# ------------Print Solution------------

# -------------Graph Solution------------
