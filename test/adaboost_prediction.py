# _*_coding:utf-8_*_
import pandas as pd
import numpy as np
from algo.Adaboost import *

# 获取训练数据集（20190301.csv）
train_data = pd.read_csv('../data/day_stock_process/20190301.csv')[['open', 'high', 'low', 'close',
                                                                    'pre_close', 'change', 'pct_chg', 'vol',
                                                                    'amount',
                                                                    'label']]
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
train_re = 0  # 训练集分正确的样本个数
for i in range(m):
    if predictions[i] == train_Y[i]:
        train_re += 1
train_acc = train_re / m
print(f'训练集准确率为{train_acc}')

"""
利用上面训练得到的分类器对新的样本数据集进行预测分类
"""

# 获取测试数据集（20190329）
test_data = pd.read_csv('../data/day_stock_process/20190301.csv')[['ts_code', 'open', 'high', 'low', 'close',
                                                                   'pre_close', 'change', 'pct_chg', 'vol',
                                                                   'amount',
                                                                   'label']]
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
print(f'测试集准确率为{test_acc}')

test_data['prediction'] = predictions
test_data['aggClass'] = aggClass
# 按分类结果降序排序
test_data.sort_values(by='prediction', ascending=False, inplace=True)
# 按分类的累计类别估计值降序排序
test_data.sort_values(by='aggClass', ascending=False, inplace=True)
# 获取前50支股票
recommend_stock = test_data.head(50)['ts_code']
print(recommend_stock.iloc[0])

# 根据股票代码获取股票名称
df = pd.read_csv('../data/stock_basic/stock_basic.csv')
dname = df['name']
dcode = df['ts_code']
stock_name = []
for i in range(len(recommend_stock)):
    name_index = dcode[dcode.values == recommend_stock.iloc[i]].index[0]
    stock_name.append(dname[name_index])

print('推荐买入的股票如下：')
print(stock_name)
