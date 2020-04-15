# _*_coding:utf-8_*_
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import Arg

args = Arg()

"""
以涨幅度为依据，给单支股票加标签
参数：
    code:股票代码
返回值：
    添加了标签的股票数据
"""


def addLabelToSingleStock(code):
    data = pd.read_csv(args.hs300path + code + '.csv')  # 读取数据
    data = data[['open', 'low', 'high', 'close', 'volume', 'p_change']].copy()
    m = data.shape[0]
    data['label'] = np.zeros((m, 1))  # 初始化标签列 0

    for i in range(m):
        p_change = data['p_change'].iloc[i]
        if p_change > 0:
            data['label'].iloc[i] = 1
        else:
            data['label'].iloc[i] = -1
    data.drop(['p_change'], axis=1)
    return data


"""
划分训练集和测试集
参数：
    dataMax:股票数据集
返回值：
    X_train：训练集特征矩阵
    Y_train：训练集标签矩阵
    X_test：测试集特征矩阵
    Y_test: 测试集标签矩阵
"""


def split_train_test(dataMax):
    # X:特征列，Y:标签列
    X = dataMax.iloc[:, :-1].values
    Y = dataMax.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # 将数据集转化为矩阵
    X_train = np.mat(X_train)
    X_test = np.mat(X_test)
    Y_train = np.mat(Y_train).T
    Y_test = np.mat(Y_test).T
    return X_train, Y_train, X_test, Y_test


"""
从已下载的数据中获取指定时间点的股票数据，并合并成一个dataframe
添加标签
"""


def merge_day_data(time):
    # 读取股票基本信息文件，获取股票代码列表
    pool = pd.read_csv('../data/stock_basic/stock_basic.csv')
    day_stock = pd.DataFrame()
    for code in pool.ts_code:
        path = '../data/mark_yield/' + code + '.csv'
        df = pd.read_csv(path, index_col='trade_date', parse_dates=True)
        df = df.sort_values('trade_date')
        df = df[['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'yield']]
        data = df[time:time]
        if data is not None:
            day_stock = pd.concat([day_stock, data])
    day_stock = day_stock.sort_values(by='yield', ascending=False)
    l = len(day_stock)
    day_stock = pd.concat([day_stock[:int(0.25 * l)], day_stock[int(0.75 * l):]])
    m = day_stock.shape[0]
    day_stock['label'] = np.ones((m, 1))
    for i in range(m):
        if i > len(day_stock) / 2 - 1:
            day_stock['label'].iloc[i] = -1
    return day_stock


"""
根据上交所交易日历，每隔20个交易日获取已下载的股票数据到指定文件中
"""


def download_time_set():
    # 获取上交所交易日期
    cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    # 以20个交易日为时间间隔，取当前时间的前十四个时间点
    cal_date = cal_date[cal_date.is_open == 1]['cal_date'][::20][14:29]
    for dt in cal_date:
        df = merge_day_data(str(dt))
        path = '../data/day_stock_process/' + str(dt) + '.csv'
        df.to_csv(path)


"""
标注股票未来20日的收益率,并存入文件中
"""


def mark_stock_yield():
    pool = pd.read_csv('../data/stock_basic/stock_basic.csv')
    for code in pool.ts_code:
        df = pd.read_csv('../data/stock_basic/' + code + '.csv')
        m = df.shape[0]
        df['yield'] = np.zeros((m, 1))
        df['yield'] = np.round((df['close'].shift(-20) - df['close']) / df['close'], 2)
        df.to_csv('../data/mark_yield/' + code + '.csv', index=0)


"""
标注沪深300指数数据未来20天的收益率
"""


def mark_hs300_yield():
    df = pd.read_csv('../data/index_data/399300.csv')
    m = df.shape[0]
    df['yield'] = np.zeros((m, 1))
    df['yield'] = np.round((df['close'] - df['close'].shift(20)) / df['close'], 2)
    df.to_csv('../data/index_data/399300.csv', index=0)
