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

from pandas_datareader.data import Options
from pricingengine import *
from pandas_datareader import data, wb
from pinance import Pinance
from dateutil.relativedelta import relativedelta
import csv


def main():
    # ----------------pre-defined parameters---------------------

    stock_name = 'TWTR'
    data_source_price = 'yahoo'
    data_source_dividend= 'yahoo-dividends'
    maturity_string_test = "2019-03-15"
    maturity_test = datetime.datetime.strptime(maturity_string_test, "%Y-%m-%d")
    n_test = 10  # step of binimial tree
    IS_PRINT = 1

    # ----------------user-defined product data---------------------

    pricing_date_string_test = "2018-10-31"
    pricing_date_test = datetime.datetime.strptime(pricing_date_string_test, "%Y-%m-%d")  # "2018-09-30"

    cp_test = CallPutType.put
    exercise_european = ExerciseType.european
    exercise_american = ExerciseType.american
    k_test = 20.0

    # ----------------market data---------------------

    # Minimum 1 year of historical data
    historical_start_date2 = pricing_date_test - (maturity_test - pricing_date_test)
    historical_start_date1 = pricing_date_test - relativedelta(years=1)
    historical_start_date_test = min(historical_start_date1, historical_start_date2)

    stock_price_list = data.DataReader(name=stock_name, data_source=data_source_price, start=historical_start_date_test, end=pricing_date_test)
    s_history_test = stock_price_list['Adj Close']
    s_test = stock_price_list['Adj Close'][pricing_date_test]

    dividend_list = data.DataReader(name=stock_name, data_source=data_source_dividend, start=historical_start_date1, end=pricing_date_test)
    if not dividend_list.empty:
        b_test = sum(list(dividend_list['value']))/s_test
    else:
        b_test = 0

    # Broken Link for Option Quote
    # getOptions = Options('AAPL')
    # all_data = getOptions.get_all_data()

    # Daily Treasury Yield Curve Rates for Risk Free Rate, Data source:
    # https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldYear&year=2018
    csv_file = open("RiskFreeRate_"+pricing_date_string_test+".csv", "r")
    reader = csv.reader(csv_file)

    risk_free_rate_x = []
    risk_free_rate_y = []
    for item in reader:
        # Drop first line
        if reader.line_num == 1:
            continue
        risk_free_rate_x.append(float(item[0]))
        risk_free_rate_y.append(float(item[1]))

    risk_free_rate = [risk_free_rate_x, risk_free_rate_y]
    csv_file.close()

    stock = Pinance(stock_name)
    if cp_test == CallPutType.call:
        stock.get_options(maturity_string_test, 'C', k_test)
    else:
        stock.get_options(maturity_string_test, 'P', k_test)
    market_price_today = stock.options_data['lastPrice']

    # ----------------performing pricing---------------------

    option_european = Option(stock_name, k_test, maturity_test, cp_test, exercise_european)

    option_european1 = Option(stock_name, 11, maturity_test, cp_test, exercise_european)
    option_european2 = Option(stock_name, 9, maturity_test, cp_test, exercise_european)
    option_european3 = Option(stock_name, 10.5, maturity_test, cp_test, exercise_european)

    options_test = [option_european1, option_european2, option_european3]
    market_price_list_test = [0.5, 2, 1]

    pricing_engine_european_bi = OptionPricingEngine(pricing_date_test, option_european)

    # ----------------printing results---------------------

    npv = pricing_engine_european_bi.npv(s_test, risk_free_rate, b_test, s_history_test, n_test, options_test, market_price_list_test)

    print("The price of European option by binomial tree is %f" % npv[0])

    print("options data :  ", stock.options_data)

    forward = Forward(stock_name, k_test, maturity_test)
    pricing_engine_forward = ForwardPricingEngine(pricing_date_test, forward)
    print("The price of forward is %f" % pricing_engine_forward.npv(s_test, risk_free_rate, b_test))

    # h0_tree, h1_tree, s_tree, bond_tree, option_tree

    # print Hedging strategy in csv files
    if len(npv) > 1 and IS_PRINT:
        with open("[ouput]h0_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[1]:
                result_writer.writerow(line)
        with open("[ouput]h1_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[2]:
                result_writer.writerow(line)
        with open("[ouput]s_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[3]:
                result_writer.writerow(line)
        with open("[ouput]bond_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[4]:
                result_writer.writerow(line)
        with open("[ouput]option_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[5]:
                result_writer.writerow(line)

if __name__ == "__main__":
    main()
