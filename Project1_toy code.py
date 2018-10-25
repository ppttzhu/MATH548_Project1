# _________________LIBRARIES_________________
import numpy as np
import matplotlib.pyplot as plt  # for plotting graphs
import math
from scipy.stats import norm


# ________________FUNCTIONS______________________

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
    Black–Scholes formula support function
    """
    d1 = (math.log(s / k) + (r + sigma * sigma / 2) * t) / sigma * math.sqrt(t)

    return d1


def bs_formula_d2(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Black–Scholes formula support function
    """
    d2 = (math.log(s / k) + (r - sigma * sigma / 2) * t) / sigma * math.sqrt(t)

    return d2

# ------------Data Retrieval & Prep------------

# ------------Inputs & Outputs------------

# ------------Print Solution------------

# -------------Graph Solution------------
