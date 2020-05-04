#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: Trace
Date: 2020/5/2 14:17
@Description: 一次预测情况。当前时间往前的前240个交易日数据，每隔20个交易日取样一次，合并这些时间点的数据作为训练数据集，使用最新一期的数据集
              作为测试集，预测股票未来一段时间的收益率，并在下一个交易日进行买入
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

from algo.Adaboost import AdaboostTrainDS, AdaClassify


def build_train_data_set():
    # 获取上交所交易日期
    cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    # 以20个交易日为时间间隔，取当前时间的前十二个时间点
    cal_date = cal_date[cal_date.is_open == 1]['cal_date'][::20][-14:-2]
    df = pd.read_csv('../data/day_stock_process/' + str(cal_date.iloc[0]) + '.csv')
    for dt in cal_date[1:]:
        path = '../data/day_stock_process/' + str(dt) + '.csv'
        data = pd.read_csv(path)
        df = pd.concat([df, data])

    price_feature = ['open', 'high', 'low', 'close', 'pre_close']
    for feature in price_feature:
        # 数据归一化
        df[feature] = minmax_scale(df[feature])
    df = df[['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'label']]
    df.to_csv('../data/train_test_set/train_mix.csv', index=False)


def build_test_data_set():
    # 获取上交所交易日期
    cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    cal_date = cal_date[cal_date.is_open == 1]['cal_date'][::20][-2:-1]
    df = pd.read_csv('../data/day_stock_process/' + str(cal_date.iloc[0]) + '.csv')
    price_feature = ['open', 'high', 'low', 'close', 'pre_close']
    for feature in price_feature:
        # 数据归一化
        df[feature] = minmax_scale(df[feature])
    df = df[['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'label']]
    df.to_csv('../data/train_test_set/test.csv', index=False)


def predict():
    # 获取训练数据集
    train_data = pd.read_csv('../data/train_test_set/train_mix.csv')
    # 训练集特征样本
    train_X = np.mat(train_data.iloc[:, :-1].values)

    # 训练集标签样本
    train_Y = np.mat(train_data.iloc[:, -1].values).T

    # 训练得到弱分类器信息
    weakClass, aggClass = AdaboostTrainDS(train_X, train_Y, maxC=40)

    # 使用弱分类器对特征矩阵进行分类
    predictions, aggClass = AdaClassify(train_X, weakClass)

    # print(predictions)
    # 计算训练集分类准确率
    m = train_X.shape[0]
    print(m)
    train_re = 0  # 训练集分正确的样本个数
    for i in range(m):
        if predictions[i] == train_Y[i]:
            train_re += 1
    train_acc = train_re / m
    print(train_re)
    print(f'训练集准确率为{train_acc}')

    """
    利用上面训练得到的分类器对新的样本数据集进行预测分类
    """

    # 获取测试数据集（20190329）
    test_data = pd.read_csv('../data/train_test_set/test.csv')
    # 测试集特征样本
    test_X = np.mat(test_data.iloc[:, 1:-1].values)

    # 测试集标签样本
    test_Y = np.mat(test_data.iloc[:, -1].values).T

    # 使用弱分类器对特征矩阵进行分类
    predictions, aggClass = AdaClassify(test_X, weakClass)

    # 计算测试集分类准确率
    test_re = 0
    n = test_X.shape[0]
    print(n)
    for i in range(n):
        if predictions[i] == test_Y[i]:
            test_re += 1
    print(test_re)
    test_acc = test_re / n
    print(f'测试集准确率为{test_acc}')

    test_data['prediction'] = predictions
    test_data['aggClass'] = aggClass
    # 按分类结果降序排序
    test_data.sort_values(by='prediction', ascending=False, inplace=True)
    # 按分类的累计类别估计值降序排序
    test_data.sort_values(by='aggClass', ascending=False, inplace=True)
    # 获取前50支股票
    recommend_stock = test_data.head(50)['ts_code']

    # 根据股票代码获取股票名称
    df = pd.read_csv('../data/stock_list/stock_basic.csv')
    dname = df['name']
    dcode = df['ts_code']
    stock_name = []
    for i in range(len(recommend_stock)):
        name_index = dcode[dcode.values == recommend_stock.iloc[i]].index[0]
        stock_name.append(dname[name_index])

    print('推荐买入的五十支股票如下：')
    print(stock_name)


if __name__ == '__main__':
    build_train_data_set()
    build_test_data_set()
    predict()
