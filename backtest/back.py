#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: Trace
Date: 2020/6/20 19:24
@Description: None
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import KFold

from algo.Adaboost import AdaboostTrainDS, AdaClassify
from util.get_model_param import predict_cross_validation

"""
构建训练集
"""


def build_train_data_set(s):
    cal_date = pd.read_csv('../data/trade_cal/date_isopen.csv')['cal_date'][::20][s:11 + s]
    df = pd.read_csv('../data/day_stock_process/' + str(cal_date.iloc[0]) + '.csv')
    for dt in cal_date[1:]:
        path = '../data/day_stock_process/' + str(dt) + '.csv'
        data = pd.read_csv(path)
        df = pd.concat([df, data])
    df = df[['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'label']]
    df.to_csv('../data/back/train_mix_' + str(s + 1) + '.csv', index=False)


"""
构建测试集
"""


def build_test_data_set(s):
    cal_date = pd.read_csv('../data/trade_cal/date_isopen.csv')['cal_date'][::20][s + 12:s + 13]
    df = pd.read_csv('../data/day_stock_process/' + str(cal_date.iloc[0]) + '.csv')
    return df


def predict(s):
    # 获取训练数据集
    train_data = pd.read_csv('../data/back/train_mix_' + str(s + 1) + '.csv')
    # 训练集特征样本
    train_X = np.mat(train_data.iloc[:, :-1].values)

    # 训练集标签样本
    train_Y = np.mat(train_data.iloc[:, -1].values).T

    # 训练次数
    maxC = 8
    accuracy = 0
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    for i in range(1, 50):
        for train_index, test_index in kf.split(train_data):
            test_acc = predict_cross_validation(train_data.iloc[train_index], train_data.iloc[test_index], i)
            if accuracy < test_acc:
                accuracy = test_acc
                maxC = i
    print(f'弱分类器个数：{maxC}')
    # 训练得到弱分类器信息
    weakClass, aggClass = AdaboostTrainDS(train_X, train_Y, maxC)

    # 使用弱分类器对特征矩阵进行分类
    predictions, aggClass = AdaClassify(train_X, weakClass)

    # 计算训练集分类准确率
    m = train_X.shape[0]

    train_re = 0  # 训练集分正确的样本个数
    for i in range(m):
        if predictions[i] == train_Y[i]:
            train_re += 1
    train_acc = train_re / m

    # print(f'训练集准确率为{train_acc}')

    """
    利用上面训练得到的分类器对新的样本数据集进行预测分类
    """

    # 获取测试数据集
    test_data = build_test_data_set(s)

    # 测试集特征样本
    test_X = np.mat(test_data.iloc[:, 1:-1].values)

    # 测试集标签样本
    test_Y = np.mat(test_data.iloc[:, -1].values).T

    # 使用弱分类器对特征矩阵进行分类
    predictions, aggClass = AdaClassify(test_X, weakClass)

    # 计算测试集分类准确率
    test_re = 0
    n = test_X.shape[0]
    for i in range(n):
        if predictions[i] == test_Y[i]:
            test_re += 1
    test_acc = test_re / n
    # print(f'测试集准确率为{test_acc}')

    test_data['prediction'] = predictions
    test_data['aggClass'] = aggClass
    # 按分类结果降序排序
    test_data.sort_values(by='prediction', ascending=False, inplace=True)
    # 按分类的累计类别估计值降序排序
    test_data.sort_values(by='aggClass', ascending=False, inplace=True)
    # 获取前50支股票
    recommend_stock = test_data.head(50)['ts_code']
    return recommend_stock


def back_test():
    # for i in range(15):
    #     build_train_data_set(i)
    # 起始资金1000万
    capital_base = 10000000
    # 买卖后的资金
    capital_change = capital_base
    # 持仓股票代码和每只股票数量及价格
    keepstock = DataFrame({'code': [], 'number': [], 'price': []})
    # 存储日期和收益
    save_date_yield = DataFrame({'date': [], 'yield': []})
    for s in range(15):
        recommend_stock = predict(s)
        cal_date = pd.read_csv('../data/trade_cal/date_isopen.csv')['cal_date'][::20][s + 12:s + 13]
        test_data = pd.read_csv('../data/day_stock_process/' + str(cal_date.iloc[0]) + '.csv')
        dcode = test_data['ts_code']

        # ------------------- 卖出股票-------------------------
        if not keepstock.empty:
            for stock in keepstock['code'].values:
                if stock not in recommend_stock:
                    stock_index = keepstock.loc[(keepstock['code'] == stock)].index[0]
                    # 股票数量
                    number = keepstock['number'][stock_index]
                    # 获取股票价格
                    if stock in dcode:
                        name_index = dcode[dcode == stock].index[0]
                        price = test_data['close'][name_index]
                    else:
                        price_index = keepstock.loc[(keepstock['code'] == stock)].index[0]
                        price = keepstock['price'][price_index]
                    # 计算剩余资金
                    capital_change += number * price
                    # 删除股票
                    keepstock = keepstock.drop(stock_index)
        # ------------------- 买入股票-------------------------
        i = 0
        if keepstock.empty:
            for stock in recommend_stock:
                name_index = dcode[dcode == stock].index[0]
                price = test_data['open'].iloc[name_index]
                # 买入
                capital_change -= 100000
                # 计算能买多少股
                num = int(100000 / price)
                i += 1
                keepstock = keepstock.append({'code': stock, 'number': num, 'price': price}, ignore_index=True)

        else:
            for stock in recommend_stock:
                name_index = dcode[dcode == stock].index[0]
                price = test_data['open'][name_index]
                if stock not in keepstock['code'].values:
                    # 买入
                    capital_change -= 100000
                    # 计算能买多少股
                    num = int(100000 / price)
                    i += 1
                    keepstock = keepstock.append({'code': stock, 'number': num, 'price': price}, ignore_index=True)
                else:
                    keepstock.loc[name_index, 'price'] = price
        if not keepstock.empty:
            for stock in keepstock['code'].values:
                # 股票价格
                if stock in dcode:
                    name_index = dcode[dcode == stock].index[0]
                    price = test_data['close'][name_index]
                else:
                    price_index = keepstock.loc[(keepstock['code'].values == stock)].index[0]
                    price = keepstock['price'][price_index]
                # 股票数量
                index = keepstock.loc[(keepstock['code'] == stock)].index[0]
                number = keepstock['number'][index]
                capital_change += price * number
        rate = np.round((capital_change - capital_base) / capital_base, 2)
        # rate = (capital_change - capital_base) / capital_base
        capital_base = capital_change
        save_date_yield = save_date_yield.append({'date': str(cal_date.iloc[0]), 'yield': rate}, ignore_index=True)
    save_date_yield.to_csv("../data/back/result.csv")


"""
计算沪深300收益率曲线图
"""


def calhs300yearprofit():
    # 起始资金1000万
    capital_base = 10000000
    capital_change = capital_base
    # 持股
    keepstocknum = 0
    # 存储日期和收益率
    save_date_yield = DataFrame({'date': [], 'yield': []})
    # 读取沪深300指数数据
    df = pd.read_csv('../data/index_data/399300.csv')
    data = df[['date', 'open', 'close']][1::20]
    for dt in data['date']:
        # 获取开盘价和收盘价
        dt_index = data['date'][data['date'] == dt].index[0]
        open_price = data['open'][dt_index]
        close_price = data['close'][dt_index]
        # 卖出持有的股票
        capital_change += keepstocknum * close_price
        # 计算收益率
        rate = np.round((capital_change - capital_base) / capital_base, 2)
        capital_base = capital_change
        save_date_yield = save_date_yield.append({'date': str(dt), 'yield': rate}, ignore_index=True)
        # 买入5000000元的股票
        capital_change -= 5000000
        # 多少股
        keepstocknum = int(5000000 / open_price)
    return save_date_yield


if __name__ == '__main__':
    back_test()

