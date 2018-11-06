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
from pandas_datareader import data
from pinance import Pinance
from dateutil.relativedelta import relativedelta
import csv


def main():
    # ----------------pre-defined parameters---------------------

    stock_name = 'TWTR'
    data_source_price = 'yahoo'
    data_source_dividend = 'yahoo-dividends'
    maturity_string_test = "2019-03-15"
    maturity_test = datetime.datetime.strptime(maturity_string_test, "%Y-%m-%d")
    pricing_date_string_test = "2018-11-05"
    pricing_date_test = datetime.datetime.strptime(pricing_date_string_test, "%Y-%m-%d")
    IS_CALCULATOR = 0  # use a calculator or not

    # ----------------product data---------------------

    # Get options for calibration, source: bbg
    csv_file = open("[input]Options_for_calibrate_" + pricing_date_string_test + ".csv", "r")
    reader = csv.reader(csv_file)

    options_for_calibrate_list = []
    options_for_calibrate_price_list = []
    options_list = []
    options_price_list = []

    for item in reader:
        # Drop first line
        if reader.line_num == 1:
            continue

        maturity_date = datetime.datetime.strptime(item[2], "%Y-%m-%d")
        option = Option(item[0], float(item[1]), maturity_date, CallPutType(int(item[3])), ExerciseType(int(item[4])))
        options_for_calibrate_list.append(option)
        options_for_calibrate_price_list.append(float(item[5]))
        if item[2] == maturity_string_test:
            options_list.append(option)
            options_price_list.append(float(item[5]))

    csv_file.close()

    if IS_CALCULATOR:
        print("windows")

    # ----------------market data---------------------

    # Minimum 1 year of historical data
    historical_start_date2 = pricing_date_test - (maturity_test - pricing_date_test)
    historical_start_date1 = pricing_date_test - relativedelta(years=1)
    historical_start_date_test = min(historical_start_date1, historical_start_date2)

    stock_price_list = data.DataReader(name=stock_name, data_source=data_source_price, start=historical_start_date_test,
                                       end=pricing_date_test)
    s_history_test = stock_price_list['Adj Close']
    s_test = stock_price_list['Adj Close'][pricing_date_test]

    dividend_list = data.DataReader(name=stock_name, data_source=data_source_dividend, start=historical_start_date1,
                                    end=pricing_date_test)
    if not dividend_list.empty:
        b_test = sum(list(dividend_list['value'])) / s_test
    else:
        b_test = 0

    # Daily Treasury Yield Curve Rates for Risk Free Rate, Data source:
    # https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldYear&year=2018
    csv_file = open("[input]RiskFreeRate_" + pricing_date_string_test + ".csv", "r")
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

    if IS_CALCULATOR:
        stock = Pinance(stock_name)
        if options_list[0].call_put_type == CallPutType.call:
            stock.get_options(maturity_string_test, 'C', options_list[0].strike)
        else:
            stock.get_options(maturity_string_test, 'P', options_list[0].strike)
        market_price_today = stock.options_data['lastPrice']

    # ----------------performing pricing---------------------

    pricing_engine = OptionPricingEngine(pricing_date_test, options_list[0])
    parameters = pricing_engine.calibrate(s_test, risk_free_rate, b_test, s_history_test, options_for_calibrate_list,
                                          options_for_calibrate_price_list)

    npv_list = []
    for option in options_list:
        pricing_engine = OptionPricingEngine(pricing_date_test, option)
        npv = pricing_engine.npv(s_test, risk_free_rate, b_test, parameters[0], parameters[1])
        npv_list.append(npv)

    # ----------------printing results---------------------

    if not parameters[0] == []:
        print("sigma = %f" % parameters[0])
    if parameters[1]:
        print("up = %f" % parameters[1][0])
        print("down = %f" % parameters[1][1])
        print("q_up = %f" % parameters[1][2])
        print("q_down = %f" % parameters[1][3])

    if not IS_CALCULATOR:
        with open("[output]npv_list.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for i in range(len(options_list)):
                result_writer.writerow([options_list[i].product_id, npv_list[i][0], options_price_list[i],
                                        npv_list[i][0] - options_price_list[i]])

    if IS_CALCULATOR:
        print("price = " + npv[0])

    # export hedging strategy in csv files
    if len(npv) > 1 and IS_CALCULATOR:
        with open("[output]h0_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[1]:
                result_writer.writerow(line)
        with open("[output]h1_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[2]:
                result_writer.writerow(line)
        with open("[output]s_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[3]:
                result_writer.writerow(line)
        with open("[output]bond_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[4]:
                result_writer.writerow(line)
        with open("[output]option_tree.csv", "w", newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar=',')
            for line in npv[5]:
                result_writer.writerow(line)

    # forward = Forward(stock_name, k_test, maturity_test)
    # pricing_engine_forward = ForwardPricingEngine(pricing_date_test, forward)
    # print("The price of forward is %f" % pricing_engine_forward.npv(s_test, risk_free_rate, b_test))


if __name__ == "__main__":
    main()
