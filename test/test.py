import datetime

import pandas as pd
import time
import tushare as ts
from config import Arg
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame

from backtest.backtest import calStockYearProfit, drawprofitline
from draw import draw_hs300_index
from util.data_utils import download_all_stock, get_hs300_data, update_stock_data
from util.dataprocess import mark_stock_yield, merge_day_data, download_time_set, mark_hs300_yield

pro = ts.pro_api('769b6990fd248e065e95887933ea517ae21e8dacdbd24bc0d1cf673a')
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


if __name__ == '__main__':
    # get_hs300_data('2018-01-01','2020-04-15','../data/index_data/')
    # mark_hs300_yield()
    # draw_hs300_index()
    # download_all_stock()
    # mark_stock_yield()
    # download_time_set()
    # calYearProfit()
    # df = merge_day_data('20180102')
    # path = '../data/back/20180102.csv'
    # df.to_csv(path)
    # drawhs300line()

    # a = DataFrame({'name':[],'age':[]})
    # # a = a.append({'name':'zs','age':12},ignore_index=True)
    # if not a.empty:
    #     if 'ls' not in a['name'].values:
    #          print("hello")
    #     else:
    #         print("none")
    #         print(type(a['name']))
    # else:
    #     print("kong")
    # df = calStockYearProfit()
    # df.to_csv('../data/back/result.csv')
    # drawprofitline()

    # for t in ['20190301','20190329','20190429','20190530','20190628','20190726','20190823','20190923','20191028','20191125','20191223','20200121','20200226','20200325']:
    #     df = pro.daily(trade_date=t)
    #     print(len(df))

    # df = pd.read_csv('../data/hs300/000001.csv', index_col='date', parse_dates=True)
    # print(df)
    # t = time.strftime("%Y%m%d",time.localtime(time.time()))
    # print(t)
    # print(type(t))
    # arg = Arg()
    # print(arg.current)
    # print(type(arg.current))
    # download_all_stock()
    # update_stock_data()
    import math

    df = pd.read_csv('../data/data_from_mongodb/600000.csv')

    df.dropna(subset=['totalMarketCapitalization'], inplace=True)

    # 取对数
    df['totalMarketCapitalization'] = df['totalMarketCapitalization'].apply(lambda x: math.log(x))
    # print(df['totalMarketCapitalization'])

    # 画箱线图

    # 设置图形的显示风格
    plt.style.use('ggplot')

    # 设置中文和负号正常显示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 绘图：整体乘客的年龄箱线图

    # plt.boxplot(x=df.totalMarketCapitalization,  # 指定绘图数据
    #
    #             patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充
    #
    #             showmeans=True,  # 以点的形式显示均值
    #
    #             boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色
    #
    #             flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # 设置异常值属性，点的形状、填充色和边框色
    #
    #             meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色
    #
    #             medianprops={'linestyle': '--', 'color': 'orange'})  # 设置中位数线的属性，线的类型和颜色
    #
    #
    # plt.show()

    def filter_extreme_MAD(series,n):
        """
        :param series:
        :param n:
        :return:中位数去极值
        """
        median = np.percentile(series,50)
        new_median = np.percentile((series - median).abs(),50)
        max_range = median + n*new_median
        min_range = median - n*new_median
        return np.clip(series,min_range,max_range)
    df['totalMarketCapitalization'] = filter_extreme_MAD(df['totalMarketCapitalization'],5)
    # print(df['totalMarketCapitalization'])

    # 标准化
    from sklearn.preprocessing import StandardScaler
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    features = ['totalMarketCapitalization']
    df = df[features]
    df[features] = StandardScaler().fit_transform(df)
    print(df)

