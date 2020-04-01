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
    data = pd.read_csv(args.hs300path+code+'.csv')  # 读取数据
    data = data[['open', 'low', 'high', 'close', 'volume', 'p_change']].copy()
    m = data.shape[0]
    data['label'] = np.zeros((m, 1))    # 初始化标签列 0

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