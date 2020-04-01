# _*_coding:utf-8_*_
from algo.Adaboost import *
from util.dataprocess import *

"""
以平安银行为例，使用AdaBoost预测的准确率
    1、加载数据集并给数据集贴上标签
    2、切割数据集获得训练集和测试集的特征矩阵和标签矩阵
    3、训练得到弱分类器信息
    4、使用弱分类器对测试集特征矩阵进行分类
    5、计算分类准确率
"""


def calAcc(maxC=40):
    # 1、加载数据集并给数据集贴上标签
    dataMat = addLabelToSingleStock('000001')

    # 2、切割数据集获得训练集和测试集的特征矩阵和标签矩阵
    train_X, train_Y, test_X, test_Y = split_train_test(dataMat)

    # 3、训练得到弱分类器信息
    weakClass = AdaboostTrainDS(train_X, train_Y, maxC)

    # 4、使用弱分类器对特征矩阵进行分类
    predictions = AdaClassify(train_X, weakClass)

    # 5、计算训练集分类准确率
    m = train_X.shape[0]
    train_re = 0  # 训练集分正确的样本个数
    for i in range(m):
        if predictions[i] == train_Y[i]:
            train_re += 1
    train_acc = train_re / m
    print(f'训练集准确率为{train_acc}')

    # 6、计算测试集分类准确率
    test_re = 0
    n = test_X.shape[0]
    predictions = AdaClassify(test_X, weakClass)
    for i in range(n):
        if predictions[i] == test_Y[i]:
            test_re += 1
    test_acc = test_re / n
    print(f'测试集准确率为{test_acc}')
    return train_acc, test_acc

"""
测试不同的迭代次数预测的准确率
"""


def reCal():
    cycles = [1, 10, 15, 50, 100, 500, 1000]
    train_acc = []
    test_acc = []
    for maxC in cycles:
        a, b = calAcc(maxC)
        train_acc.append(round(a * 100, 2))
        test_acc.append(round(b * 100, 2))
    df = pd.DataFrame({'分类器数目': cycles, '训练集准确率': train_acc, '测试集准确率': test_acc})
    return df


if __name__ == '__main__':
    reCal()
