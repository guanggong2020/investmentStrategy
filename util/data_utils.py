# _*_coding:utf-8_*_
import tushare as ts
import pandas as pd
import os
import time


# ----------------------获取当前时间------------------- #
def getCurrentTime():
    return time.strftime('[%Y-%m-%d]', time.localtime(time.time()))


# ----------------------下载某只股票数据------------------- #
# code:股票编码 日期格式：2020-03-11 filename：文件夹路径../data/
# length是筛选股票长度，默认值为-1，即不做筛选，可人为指定长度，如300，既少于300天的股票不保存
def get_stock_data(code, date1, date2, filename, length=-1):
    df = ts.get_hist_data(code, start=date1, end=date2)
    df1 = pd.DataFrame(df)
    df1 = df1[['open', 'high', 'close', 'low', 'volume', 'p_change']]
    df1 = df1.sort_values(by='date')
    print('共有%s天数据' % len(df1))
    if not os.path.exists(filename):
        os.makedirs(filename)
    if length == -1:
        path = code + '.csv'
        df1.to_csv(os.path.join(filename, path))
    else:
        if len(df1) >= length:
            path = code + '.csv'
            df1.to_csv(os.path.join(filename, path))


# ----------------------下载沪深300指数数据------------------- #
# date1是开始日期，date2是截止日期，filename是文件存放目录
def get_hs300_data(date1, date2, filename):
    df = ts.get_hist_data('399300', start=date1, end=date2)
    df1 = pd.DataFrame(df)
    df1 = df1[['open', 'high', 'close', 'low', 'volume', 'p_change']]
    df1 = df1.sort_values(by='date')
    print('共有%s天数据' % len(df1))
    df1.to_csv(os.path.join(filename, '399300.csv'))


# ------------------------更新股票数据------------------------ #
# 将股票数据从本地文件的最后日期更新至当日
# code:股票代码
def update_stock_data(code):
    filename = '../data/' + code + '.csv'
    (filepath, tempfilename) = os.path.split(filename)
    (stock_code, extension) = os.path.splitext(tempfilename)
    f = open(filename, 'r')
    df = pd.read_csv(f)
    print('股票{}文件中的最新日期为:{}'.format(stock_code, df.iloc[-1, 0]))
    data_now = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    print('更新日期至：%s' % data_now)
    nf = ts.get_hist_data(stock_code, str(df.iloc[-1, 0]), data_now)
    nf = nf.sort_values(by='date')
    nf = nf.iloc[1:]
    print('共有%s天数据' % len(nf))
    nf = pd.DataFrame(nf)
    nf = nf[['open', 'high', 'close', 'low', 'volume', 'p_change']]
    nf.to_csv(filename, mode='a', header=False)
    f.close()


# ----------------------下载沪深300股票信息------------------- #
def get_hs300_code_name(filename):
    df = ts.get_hs300s()
    df1 = pd.DataFrame(df)
    df1 = df1[['name', 'code']]
    df1.to_csv(os.path.join(filename, 'hs300.csv'))


# ------------------------批量下载沪深300股票数据------------------------ #
def download_hs300_stock_data(date1, date2, filename):
    df = pd.read_csv('../data/hs300/hs300.csv')['code']
    for code in df:
        code = "{0:06d}".format(code)
        if not os.path.exists('data/{}.csv'.format(code)):
            get_stock_data(code, date1, date2, filename)


# ------------------------使用pandas去除重复数据------------------------ #
def quchong(file):
    f = open(file)
    df = pd.read_csv(f, header=0)
    datalist = df.drop_duplicates()
    datalist.to_csv(file)


# ------------------------获取股票长度----------------------- #
# 辅助函数
def get_data_len(file_path):
    df = pd.read_csv(file_path)
    return len(df)
