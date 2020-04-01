# _*_coding:utf-8_*_
import pandas as pd
from pandas import DataFrame
from algo.Adaboost import AdaboostTrainDS, AdaClassify
from util.data_utils import get_data_len
from util.dataprocess import loadDataSet

# 获取数据集
df = pd.read_csv('../data/hs300/hs300.csv')['code']
result = {'code':[],'accuracy':[],'result':[]}
result = DataFrame(result)
for code in df:
    code = "{0:06d}".format(code)
    length = get_data_len('../data/hs300/'+code+'.csv')
    if length > 240:
        train_X, train_Y, test_X, test_Y = loadDataSet(code)
        # 训练
        weakClass = AdaboostTrainDS(train_X, train_Y, maxC=15)
        # 预测
        predictions = AdaClassify(train_X, weakClass)
        # 训练集准确率
        m = train_X.shape[0]
        train_re = 0  # 训练集分正确的样本个数
        for i in range(m):
            if predictions[i] == train_Y[i]:
                train_re += 1
        train_acc = train_re / m
        # 测试集预测结果
        n = test_X.shape[0]
        predictions = AdaClassify(test_X, weakClass)
        re = DataFrame({'code':[code],'accuracy':[train_acc],'result':[predictions[0]]})
        result = result.append(re,ignore_index=True)

print(result)
