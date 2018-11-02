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

cp_test = CallPutType.put
exercise_european = ExerciseType.european
exercise_american = ExerciseType.american
s_test = 10.0
k_test = 10.0
maturity_test = datetime.datetime.strptime("2019-3-15","%Y-%m-%d")
pricing_date_test = datetime.datetime.strptime("2018-10-31","%Y-%m-%d")
r_test = 0.02
b_test = 0.00
sigma_test = 0.1
n_test = 100

option_european = Option("000001.SZ", k_test, maturity_test, cp_test, exercise_european)
option_american = Option("000001.SZ", k_test, maturity_test, cp_test, exercise_american)
pricing_engine_european_bi = OptionPricingEngine(pricing_date_test, PricingMethod.binomial_tree_model, option_european)
pricing_engine_american_bi = OptionPricingEngine(pricing_date_test, PricingMethod.binomial_tree_model, option_american)
pricing_engine_european_bs = OptionPricingEngine(pricing_date_test, PricingMethod.bs_baw_benchmarking_model, option_european)
pricing_engine_american_baw = OptionPricingEngine(pricing_date_test, PricingMethod.bs_baw_benchmarking_model, option_american)

print("The price of European option by binomial tree is %f" % pricing_engine_european_bi.npv(s_test, r_test, b_test, sigma_test, n_test))
print("The price of European option by BS formula is %f" % pricing_engine_european_bs.npv(s_test, r_test, b_test, sigma_test, n_test))
print("The price of American option by binomial tree is %f" % pricing_engine_american_bi.npv(s_test, r_test, b_test, sigma_test, n_test))
print("The price of American option by BAW formula is %f" % pricing_engine_american_baw.npv(s_test, r_test, b_test, sigma_test, n_test))