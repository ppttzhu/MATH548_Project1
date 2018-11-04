#!/usr/bin/env python3
# encoding: utf-8

"""
@Project: MATH548_Project1
@FileName: testing.py
@Author: Kim Ki Hyeon, Lu Weikun, Peng Yixin, Zhou Nan
@Date: 2018/10/25
@Description：
@File URL: https://github.com/ppttzhu/MATH548_Project1/import
"""

from pricingengine import *

cp_test = CallPutType.call
exercise_european = ExerciseType.european
exercise_american = ExerciseType.american
s_test = 10.0
k_test = 10.0
maturity_test = datetime.datetime.strptime("2019-3-15","%Y-%m-%d")
pricing_date_test = datetime.datetime.strptime("2018-10-31","%Y-%m-%d")
# t = 0.3698630136986301
s_history_test = [11, 12, 11, 10, 9, 8]
r_test = 0.02
b_test = 0.01
sigma_test = 0.1
n_test = 100

option_european = Option("000001.SZ", k_test, maturity_test, cp_test, exercise_european)
option_american = Option("000001.SZ", k_test, maturity_test, cp_test, exercise_american)

option_european1 = Option("000001.SZ", 11, maturity_test, cp_test, exercise_european)
option_european2 = Option("000001.SZ", 9, maturity_test, cp_test, exercise_european)
option_european3 = Option("000001.SZ", 10.5, maturity_test, cp_test, exercise_european)

options_test = [option_european1, option_european2, option_european3]
market_price = [0.5, 2, 1]

pricing_engine_european_bi = OptionPricingEngine(pricing_date_test, PricingMethod.binomial_tree_model, option_european)
pricing_engine_american_bi = OptionPricingEngine(pricing_date_test, PricingMethod.binomial_tree_model, option_american)
pricing_engine_european_bs = OptionPricingEngine(pricing_date_test, PricingMethod.bs_baw_benchmarking_model, option_european)
pricing_engine_american_baw = OptionPricingEngine(pricing_date_test, PricingMethod.bs_baw_benchmarking_model, option_american)

print("The price of European option by binomial tree is %f" % pricing_engine_european_bi.npv(s_test, r_test, b_test, s_history_test, n_test, options_test, market_price))
print("The price of European option by BS formula is %f" % pricing_engine_european_bs.npv(s_test, r_test, b_test, s_history_test, n_test, options_test, market_price))
print("The price of American option by binomial tree is %f" % pricing_engine_american_bi.npv(s_test, r_test, b_test, s_history_test, n_test, options_test, market_price))
print("The price of American option by BAW formula is %f" % pricing_engine_american_baw.npv(s_test, r_test, b_test, s_history_test, n_test, options_test, market_price))

forward = Forward("000001.SZ", k_test, maturity_test)
pricing_engine_forward = ForwardPricingEngine(pricing_date_test, forward)
print("The price of forward is %f" % pricing_engine_forward.npv(s_test, r_test, b_test))
