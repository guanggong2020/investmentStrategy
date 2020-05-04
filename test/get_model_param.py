# _*_ coding:utf-8_*_

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

from algo.Adaboost import *

from sklearn.model_selection import KFold

from util.data_utils import split_train_test

"""
获取弱分类器个数从1增加到200的过程中训练集和测试集的准确率
"""

# 获取训练数据集（20190301.csv）
train_data = pd.read_csv('../data/train_test_set/train_mix.csv')[['open', 'high', 'low', 'close',
                                                                    'pre_close', 'change', 'pct_chg', 'vol',
                                                                    'amount',
                                                                    'label']]

price_feature = ['open', 'high', 'low', 'close', 'pre_close']
for feature in price_feature:
    # 数据归一化
    train_data[feature] = minmax_scale(train_data[feature])

# 打乱数据集
num = train_data.shape[0]
data_index = np.arange(num)
np.random.shuffle(data_index)
train_data = train_data.iloc[data_index]


def calAcc(train_data, maxC=40):
    # 切割数据集获得训练集和测试集的特征矩阵和标签矩阵(训练集：测试集=0.8:0.2)
    train_X, train_Y, test_X, test_Y = split_train_test(train_data)
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

    # 计算测试集分类准确率
    test_re = 0
    n = test_X.shape[0]
    predictions, aggClass = AdaClassify(test_X, weakClass)
    for i in range(n):
        if predictions[i] == test_Y[i]:
            test_re += 1
    test_acc = test_re / n
    # print(f'测试集准确率为{test_acc}')

    return train_acc, test_acc


# 计算训练集和测试集在弱分类器个数从1增加到200的过程中的准确率
def reCal():
    train_accuracy = []
    test_accuracy = []
    for i in range(1, 201):
        train_acc, test_acc = calAcc(train_data, i)
        train_accuracy.append(round(train_acc*100,2))
        test_accuracy.append(round(test_acc*100,2))
    return train_accuracy, test_accuracy


# train_accuracy,test_accuracy = reCal()
# print(train_accuracy)
# print(test_accuracy)


"""
5折交叉验证
"""


def predict_cross_validation(trainMat, testMat, maxC=40):
    # 训练集特征样本
    train_X = np.mat(trainMat.iloc[:, :-1].values)
    # 训练集标签样本
    train_Y = np.mat(trainMat.iloc[:, -1].values).T
    # 训练得到弱分类器信息
    weakClass, aggClass = AdaboostTrainDS(train_X, train_Y, maxC)
    # 测试集特征样本
    test_X = np.mat(testMat.iloc[:, :-1].values)
    # 测试集标签样本
    test_Y = np.mat(testMat.iloc[:, -1].values).T
    # 使用弱分类器对特征矩阵进行分类
    predictions, aggClass = AdaClassify(test_X, weakClass)
    # 计算测试集分类准确率
    test_re = 0
    n = test_X.shape[0]
    for i in range(n):
        if predictions[i] == test_Y[i]:
            test_re += 1
    test_acc = test_re / n
    return test_acc


def cross_validation():
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    scores = []
    for i in range(1, 201):
        for train_index, test_index in kf.split(train_data):
            # print(train_data.iloc[train_index],train_data.iloc[test_index])
            score = []
            test_acc = predict_cross_validation(train_data.iloc[train_index], train_data.iloc[test_index], i)
            score.append(test_acc)
        scores.append(round(np.mean(score)*100,2))
    return scores


if __name__ == '__main__':
    """
    获取数据，写入文件，画图
    """
    # 训练集和测试集的准确率
    train_accuracy, test_accuracy = reCal()
    # 5折交叉验证准确率
    scores = cross_validation()
    df = pd.DataFrame({'5折交叉验证准确率': scores, '训练准确率': train_accuracy, '测试准确率': test_accuracy})
    df.to_csv('../data/accuracy/accuracy_v7.csv')

