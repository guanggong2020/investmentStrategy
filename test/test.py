import datetime

import pandas as pd
import time

from util.dataprocess import mark_stock_yield, merge_day_data, download_time_set

"""
代码测试文件
"""


# df = pd.read_csv('../data/stock_basic/000001.SZ.csv', index_col='trade_date', parse_dates=True)
#
# print('--------------------------------------------------------------')
# df = df.sort_values('trade_date')
# df = df[['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']]
# print(df.head())
# print('--------------------------------------------------------------')
# stock = pd.DataFrame()
# data = df['2020-04-07':'2020-04-07']
# if data is not None:
#     stock = pd.concat([stock, data])
# print(stock)


# def merge_day_data(time):
#     pool = pd.read_csv('../data/stock_basic/stock_basic.csv')
#     day_stock = pd.DataFrame()
#     for code in pool.ts_code:
#         path = '../data/stock_basic/' + code + '.csv'
#         df = pd.read_csv(path, index_col='trade_date', parse_dates=True)
#         df = df.sort_values('trade_date')
#         df = df[['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']]
#         data = df[time:time]
#         if data is not None:
#             day_stock = pd.concat([day_stock, data])
#     return day_stock


# if __name__ == '__main__':
    # df = pd.read_csv('../data/stock_basic/000001.SZ.csv', index_col='trade_date')
    # # print(df.index[::20][1:14])
    # t = df.index[::20][1:2]
    # for t1 in t:
    # t1='20200306'
    # df = merge_day_data(t1)
    # path = '../data/day_stock/'+str(t1)+'.csv'

    # 获取上交所交易日期
    # cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    # # 以20个交易日为时间间隔，取当前时间的前十四个时间点
    # cal_date = cal_date[cal_date.is_open == 1]['cal_date'][::20][14:29]
    # for dt in cal_date:
    #     df = merge_day_data(str(dt))
    #     path = '../data/day_stock/' + str(dt) + '.csv'
    #     df.to_csv(path)
    # print(len(cal_date))
    # print(cal_date)

    # df = pd.read_csv('../data/hs300/000001.csv',index_col='date',parse_dates=True)
    # print(df.head())
    # print('--------------------------------------------------------------')
    # data = df['2017-10-09':'2017-10-09']
    # stock = pd.DataFrame()
    # stock = pd.concat([stock,data])
    # print(stock)
    # print('--------------------------------------------------------------')
    # df = pd.read_csv('../data/hs300/000002.csv',index_col='date',parse_dates=True)
    # data = df['2017-10-09':'2017-10-09']
    # stock = pd.concat([stock,data])
    # print(stock)

    # mark_stock_yield()
    # day_stock = merge_day_data('20190301')
    # print(day_stock)
    # download_time_set()
    # a = [1,2,3]
    # import numpy as np
    # print(np.mean(a))