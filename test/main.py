#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: Trace
Date: 2020/4/28 18:21
@Desciption: None
"""
import math

from sklearn.preprocessing import minmax_scale

from backtest.back import build_test_data_set
from util.data_utils import *
from util.dataprocess import merge_day_data

if __name__ == '__main__':

    # 获取当前所有正常上市交易的股票列表
    # get_stock_basic()

    # 根据股票列表下载当前所有正常交易的股票数据，从2018年01月02日开始
    # while True:
    #     download_all_stock()

    # df = pd.read_csv("../data/trade_cal/trade_cal_sse.csv")
    # cal_date = df[df.is_open==1]["cal_date"][1::20]
    # cal_date.to_csv("../data/trade_cal/date_test.csv",index=False)

    # cal_date = pd.read_csv("../data/trade_cal/date_isopen.csv")
    # print(cal_date)
    # d = cal_date[::20]["cal_date"].iloc[11]
    # print(d)
    # cal_date = pd.read_csv('../data/trade_cal/date_isopen.csv')['cal_date'][::20][-14:-2]
    # print(cal_date)
    # cal_date = pd.read_csv("../data/trade_cal/date_test.csv")['cal_date']
    # for dt in cal_date:
    #     df = merge_day_data(str(dt))
    #     df = df[['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'label']]
    #     path = '../data/back/' + str(dt) + '.csv'
    #     df.to_csv(path)

    for i in range(1):
        # 获取测试数据集
        test_data = build_test_data_set(i)
        print(test_data.shape)
        # 测试集特征样本
        test_X = np.mat(test_data.iloc[:, 1:-1].values)

        # 测试集标签样本
        test_Y = np.mat(test_data.iloc[:, -1].values).T
        print(test_X)






