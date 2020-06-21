# _*_coding:utf-8_*_
import numpy as np

from algo.stump import buildStump, stumpClassify

"""
AdaBoost实现过程：
    1、利用buildStump()函数找到最佳的单层决策树
    2、将最佳单层决策树加入到单层决策树组
    3、计算Alpha
    4、计算新的权重向量D
    5、更新累计类别估计值
    6、如果错误率为0.0，退出循环
"""

"""
函数功能：基于单层决策树的AdaBoost训练过程
参数说明：
    xMat：特征矩阵
    yMat：标签矩阵
    maxC：最大迭代次数，默认40
返回值：
    weakClass：弱分类器信息
"""


def AdaboostTrainDS(xMat, yMat, maxC=40):
    weakClass = []  # 用于存储每次训练得到的弱分类器及其输出结果的权重
    m = xMat.shape[0]
    D = np.mat(np.ones((m, 1)) / m)  # 数据集权重初始化为1/m
    aggClass = np.mat(np.zeros((m, 1)))  # 记录每个数据点的类别估计累计值
    for i in range(maxC):
        Stump, error, bestClas = buildStump(xMat, yMat, D)  # 构建单层决策树
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))
        Stump['alpha'] = np.round(alpha, 2)  # 存储弱学习算法权重
        weakClass.append(Stump)  # 存储单层决策树
        expon = np.multiply(-1 * alpha * yMat, bestClas)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        aggClass += alpha * bestClas  # 更新累计类别估计值
        aggErr = np.multiply(np.sign(aggClass) != yMat, np.ones((m, 1)))
        errRate = aggErr.sum() / m
        if errRate == 0: break  # 误差为0，退出循环
    return weakClass,aggClass


"""
AdaBoost算法分类函数
    多个弱分类器的结果以其对应的alpha值作为权重，加权求和得到最后的结果
参数：
    data：待分类样本
    weakClass：弱分类器数组
返回值：
    AdaBoost算法分类的最终结果
"""


def AdaClassify(data, weakClass):
    dataMat = np.mat(data)
    m = dataMat.shape[0]
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(len(weakClass)):  # 遍历所有分类器，进行分类
        classEst = stumpClassify(dataMat,
                                 weakClass[i]['特征列'],
                                 weakClass[i]['阈值'],
                                 weakClass[i]['标志'])
        aggClass += weakClass[i]['alpha'] * classEst
    return np.sign(aggClass),aggClass
