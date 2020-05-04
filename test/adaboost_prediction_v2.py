# _*_ coding:utf-8 _*_
import pandas as pd
import numpy as np

# 获取数据集（20190301.csv）
from algo.Adaboost import AdaClassify, AdaboostTrainDS
from draw import plotROC
from util.data_utils import split_train_test

data = pd.read_csv('../data/day_stock_process/20190301.csv', index_col=0)[['open', 'high', 'low', 'close',
                                                                           'pre_close', 'change', 'pct_chg', 'vol',
                                                                           'amount',
                                                                           'label']]
# 打乱数据集
data_index = np.arange(data.shape[0])
np.random.shuffle(data_index)
data = data.iloc[data_index]

# 划分数据集
train_X, train_Y, test_X, test_Y = split_train_test(data)

# 练得到弱分类器信息
weakClass, aggClass = AdaboostTrainDS(train_X, train_Y, 7)
print(weakClass)
# plotROC(aggClass.T,train_Y)

# 使用弱分类器对特征矩阵进行分类
predictions, aggClass0 = AdaClassify(train_X, weakClass)


# 计算训练集分类准确率
m = train_X.shape[0]
train_re = 0  # 训练集分正确的样本个数
for i in range(m):
    if predictions[i] == train_Y[i]:
        train_re += 1
train_acc = train_re / m
train_acc = round(train_acc*100,2)
print(f'训练集准确率为{train_acc}')

# 计算测试集分类准确率
test_re = 0
TP = 0
FP = 0
FN = 0
TN = 0
n = test_X.shape[0]
predictions, aggClass1 = AdaClassify(test_X, weakClass)

for i in range(n):
    if predictions[i]==1 and test_Y[i]==1:
        TP += 1
    if predictions[i] == 1 and test_Y[i] == -1:
        FP += 1
    if predictions[i]==-1 and test_Y[i]==1:
        FN += 1
    if predictions[i]==-1 and test_Y[i]==-1:
        TN += 1
    if predictions[i] == test_Y[i]:
        test_re += 1
test_acc = test_re / n
test_acc = round(test_acc*100,2)
print(f'测试集准确率为{test_acc}')
print(f'TP:{TP}')
print(f'FP:{FP}')
print(f'FN:{FN}')
print(f'TN:{TN}')