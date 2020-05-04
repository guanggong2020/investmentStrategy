# _*_coding:utf-8_*_
import tushare as ts
import pandas as pd
import os
import time
import numpy as np

from sklearn.model_selection import train_test_split

from config import Arg

pro = ts.pro_api('769b6990fd248e065e95887933ea517ae21e8dacdbd24bc0d1cf673a')
arg = Arg()


def get_hs300_data(date1, date2, filename):
    """
    :param date1: 开始日期
    :param date2: 截止日期
    :param filename: 文件存放目录
    :return: 沪深300指数数据
    """
    df = ts.get_hist_data('399300', start=date1, end=date2)
    df1 = pd.DataFrame(df)
    df1 = df1[['open', 'high', 'close', 'low', 'volume', 'p_change']]
    df1 = df1.sort_values(by='date')
    print('共有%s天数据' % len(df1))
    df1.to_csv(os.path.join(filename, '399300.csv'))


def quchong(file):
    """
    :param file: 文件完整路径
    :return: 使用pandas去除重复数据
    """
    f = open(file)
    df = pd.read_csv(f, header=0)
    datalist = df.drop_duplicates()
    datalist.to_csv(file)


def get_data_len(file_path):
    """
    :param file_path: 股票数据文件完整路径
    :return: 股票数据集长度
    """
    df = pd.read_csv(file_path)
    return len(df)


def trade_cal_sse():
    """
    :return: 交易所交易日历数据
    """
    # exchange:交易所 SSE上交所 SZSE深交所
    df = pro.trade_cal(exchange='SSE', start_date='20180101', end_date='20200428')
    df.to_csv('../data/trade_cal/trade_cal_sse.csv')


def get_stock_basic():
    """
    :return: 获取当前所有正常上市交易的股票列表
    """
    # 查询当前所有正常上市交易的股票列表
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,fullname,market,'
                                                              'list_date')
    # 去除ST股
    df = df[~df.name.str.contains('ST')]
    filename = '../data/stock_list/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    df.to_csv('../data/stock_list/stock_basic.csv')


def download_all_stock():
    """
    :return: 下载所有股票数据
    """
    pool = pd.read_csv('../data/stock_list/stock_basic.csv')
    print('获得上市股票总数：', len(pool))
    j = 1
    for code in pool.ts_code:
        print('正在获取第%d家，股票代码%s.' % (j, code))
        j += 1
        path = '../data/stock_basic/' + code + '.csv'
        if not os.path.exists('../data/stock_basic/' + code + '.csv'):
            df = pro.daily(ts_code=code, start_date='20180101')
            df = df.sort_values(by='trade_date', ascending=True)
            df.to_csv(path, index=0)
            time.sleep(0.3)


def update_stock_data():
    """
    :return: 将股票数据从本地文件的最后日期更新至当日
    """
    pool = pd.read_csv('../data/stock_basic/stock_basic.csv')
    for code in pool.ts_code:
        filename = '../data/stock_basic/' + code + '.csv'
        (filepath, tempfilename) = os.path.split(filename)
        (stock_code, extension) = os.path.splitext(tempfilename)
        df = pd.read_csv(filename, error_bad_lines=False)
        print('股票{}文件中的最新日期为:{}'.format(stock_code, df.iloc[-1, 1]))
        nf = pro.daily(ts_code=code, start_date=str(df.iloc[-1, 1]))
        nf = df.sort_values(by='trade_date', ascending=True)
        nf = nf.iloc[1:]
        print('共有%s天数据' % len(nf))
        nf = pd.DataFrame(nf)
        nf.to_csv(filename, mode='a', header=False)


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
    """
    :param dataMax: 数据集
    :return:
    """
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


def filter_extreme_MAD(series, n):
    """
    :param series:
    :param n: 这里设置为5
    :return:中位数去极值
    """
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n * new_median
    min_range = median - n * new_median
    return np.clip(series, min_range, max_range)


def standardize_series(series):
    """
    :param series:
    :return: 标准化
    """
    std = series.std()
    mean = series.mean()
    return (series - mean) / std


if __name__ == '__main__':
    trade_cal_sse()
