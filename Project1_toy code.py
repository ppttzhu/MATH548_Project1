# _________________LIBRARIES_________________
import numpy as np
import matplotlib.pyplot as plt  # for plotting graphs
import math
from scipy.stats import norm


# ________________FUNCTIONS______________________

# Binomial tree

def binomial_tree_european_analytic(cp: int, s: float, k: float, t: float, r: float, sigma: float, n: int) -> float:
    """
    Binomial tree method to price European call and put option
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


def binomial_tree_american(cp: int, s: float, k: float, t: float, r: float, sigma: float, n: int) -> float:
    """
    Binomial tree method to price American call and put option
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

    # price by rolling back the payoff step by step
    z_t_1_discounted = []
    for time_step in range(n, -1, -1):
        z_t = []
        for branch in range(time_step + 1):
            # calculate intrinsic value
            payoff = 0
            s_t = s * math.pow(up, time_step - branch) * math.pow(down, branch)
            if cp == 1:
                payoff = call_payoff(s_t, k)
            elif cp == -1:
                payoff = put_payoff(s_t, k)

            # calculate max of Y(t-1) and Z*(t-1)
            if time_step == n:  # last step Z_T = Y_T
                z_t.append(payoff)
            else:
                # American option
                z_t.append(max(payoff, z_t_1_discounted[branch]))
                # European option
                # z_t.append(z_t_1_discounted[branch])

        # discount z_t to z_t_1_discounted
        z_t_1_discounted = []
        if time_step != 0:
            for i in range(time_step):
                z_t_1_discounted.append(math.exp(-r * delta_t) * (q_up * z_t[i] + q_down * z_t[i + 1]))
        else:
            return z_t[0]


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


# Black–Scholes formula for benchmarking European Option

def bs_formula(cp: int, s: float, k: float, t: float, r: float, b: float, sigma: float) -> float:
    """
    Black–Scholes formula to price European call and put option
    https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
    :param cp: indicator 1 for call, -1 for put
    :param s: spot price of the underlying asset
    :param k: strike price
    :param t: time to maturity (expressed in years)
    :param r: risk free rate (annual rate, expressed in terms of continuous compounding)
    :param b: dividend rate of underlying asset (annual rate)
    :param sigma: volatility of returns of the underlying asset
    :return: the price of call or put option
    """
    d1 = bs_formula_d1(s, k, t, r, b, sigma)
    d2 = bs_formula_d2(s, k, t, r, b, sigma)

    npv = cp * (norm.cdf(cp * d1) * s * math.exp(-b * t)  - norm.cdf(cp * d2) * k * math.exp(-r * t))

    return npv


# Barone-Adesi and Whaley formula for benchmarking American Option

def baw_formula(cp: int, s: float, k: float, t: float, r: float, b: float, sigma: float) -> float:
    """
    Barone-Adesi and Whaley formula to price American call and put option
    https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#American_options
    http://finance.bi.no/~bernt/gcc_prog/algoritms_v1/algoritms/node24.html
    :param cp: indicator 1 for call, -1 for put
    :param s: spot price of the underlying asset
    :param k: strike price
    :param t: time to maturity (expressed in years)
    :param r: risk free rate (annual rate, expressed in terms of continuous compounding)
    :param b: dividend rate of underlying asset (annual rate)
    :param sigma: volatility of returns of the underlying asset
    :return: the price of call or put option
    """
    # set the accuracy requirement for quadratic approximation
    tolerance = 1.0e-6  # set the accuracy requirement for quadratic approximation
    max_iterations = 500  # set the max iterations number

    # to do: add dividend
    european = bs_formula(cp, s, k, t, r, b, sigma)
    # If dividend rate is zero and call option, never early exercised
    if (b - 0.0) < tolerance and cp == 1:
        return european

    nn = 2.0 * b / (sigma * sigma)
    m = 2.0 * r / (sigma * sigma)
    k_cap = 1.0 - math.exp(-r * t)
    q = (-(nn - 1) + cp * math.sqrt(math.pow((nn - 1), 2.0) + (4 * m / k_cap))) * 0.5

    # seed value from paper
    qu = 0.5 * ((-nn - 1.0) + cp * math.sqrt(math.pow((nn - 1), 2.0) + 4.0 * m))
    su = k / (1.0 - 1.0 / qu)
    h2 = - (b * t + cp * 2.0 * sigma * math.sqrt(t)) * (k / (su - k))
    s_seed = k + (su - k) * (1.0 - math.exp(h2))

    # Using Newton Raphson algorithm to find critical price Si
    no_iterations = 0
    si = s_seed
    g = 1
    gprime = 1.0

    while math.fabs(g) > tolerance and math.fabs(gprime) > tolerance and no_iterations < max_iterations and si > 0.0:
        e = bs_formula(cp, si, k, t, r, b, sigma)
        d1 = bs_formula_d1(si, k, t, r, b, sigma)
        g = cp * (si - k - (1.0 / q) * si * (1 - math.exp((b - r) * t) * norm.cdf(cp * d1))) - e
        gprime = cp * (1.0 - 1.0 / q) * (1.0 - math.exp((b - r) * t) * norm.cdf(cp * d1)) \
            + (1.0 / q) * math.exp((b - r) * t) * norm.pdf(cp * d1) * (1.0 / (sigma * math.sqrt(t)))
        si = si - (g / gprime)
        no_iterations = no_iterations + 1

    if math.fabs(g) > tolerance:
        s_star = s_seed  # did not converge
    else:
        s_star = si

    if s * cp >= s_star * cp:
        american = (s - k) * cp
    else:
        d1 = bs_formula_d1(si, k, t, r, b, sigma)
        a = (cp * s_star / q) * (1.0 - math.exp((b - r) * t) * norm.cdf(cp * d1))
        american = european + a * math.pow((s / s_star), q)

    return max(american, european)


# Supporting functions

def bs_formula_d1(s: float, k: float, t: float, r: float, b: float, sigma: float) -> float:
    """
    private Black–Scholes formula support function
    """
    d1 = (math.log(s / k) + (r - b + sigma * sigma / 2) * t) / sigma * math.sqrt(t)

    return d1


def bs_formula_d2(s: float, k: float, t: float, r: float, b: float, sigma: float) -> float:
    """
    private Black–Scholes formula support function
    """
    d2 = (math.log(s / k) + (r - b - sigma * sigma / 2) * t) / sigma * math.sqrt(t)

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
cp_test = -1
s_test = 10.0
k_test = 10.0
t_test = 1.0
r_test = 0.01
b_test = 0.01
sigma_test = 0.1
n_test = 100

binomial_tree_european_analytic11 = binomial_tree_european_analytic(cp_test, s_test, k_test, t_test, r_test, sigma_test,
                                                                    n_test)
bs_formula11 = bs_formula(cp_test, s_test, k_test, t_test, r_test, b_test, sigma_test)
binomial_tree_american11 = binomial_tree_american(cp_test, s_test, k_test, t_test, r_test, sigma_test, n_test)
baw_formula11 = baw_formula(cp_test, s_test, k_test, t_test, r_test, b_test, sigma_test)

print("The price of European option by binomial tree is %f", binomial_tree_european_analytic11)
print("The price of European option by BS formula is %f", bs_formula11)
print("The price of American option by binomial tree is %f", binomial_tree_american11)
print("The price of American option by BAW formula is %f", baw_formula11)
