# _*_coding:utf-8_*_
import numpy as np

"""
单层决策树分类函数
    通过阈值比较对数据进行分类
    参数：
        dataMatrix: 数据矩阵
        i: 第i列，也就是第几个特征
        Q: 阈值
        S: 标志，决定了不等号是大于还是小于
"""


def stumpClassify(dataMatrix, i, Q, S):
    retArray = np.ones((dataMatrix.shape[0], 1))  # 将retArray全部初始化为1
    # 根据阈值和不等号将满足要求的都设为-1
    if S == 'lt':
        retArray[dataMatrix[:, i] <= Q] = -1  # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:, i] > Q] = -1  # 如果大于阈值,则赋值为-1
    return retArray


"""
找到数据集上最佳的单层决策树
参数说明:
    xMat：特征矩阵
    yMat：标签矩阵
    D：样本权重,用于计算加权错误率
返回值:
    bestStump：最佳单层决策树信息
    minE：最小误差
    bestClas：最佳的分类结果
"""


def buildStump(xMat, yMat, D):
    xMat = np.mat(xMat)
    yMat = np.mat(yMat)
    m, n = xMat.shape  # m为样本个数(行数)，n为特征数（列数）
    Steps = 10  # 初始化一个步数
    bestStump = {}  # 用字典形式来储存树桩信息
    bestClas = np.mat(np.zeros((m, 1)))  # 初始化分类结果为0
    minE = np.inf  # 最小误差初始化为正无穷大

    for i in range(n):  # 第一层循环，遍历所有特征
        Min = xMat[:, i].min()  # 找到特征中最小值
        Max = xMat[:, i].max()  # 找到特征中最大值
        stepSize = (Max - Min) / Steps  # 计算步长

        for j in range(-1, int(Steps) + 1):  # 第二层循环，遍历每个步长

            for S in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                Q = (Min + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(xMat, i, Q, S)  # 根据阈值和不等号进行分类
                err = np.mat(np.ones((m, 1)))  # 初始化误差矩阵，先假设所有的结果都是错的（标记为1）
                err[predictedVals == yMat] = 0  # 分类正确的,赋值为0
                eca = D.T * err  # 计算误差
                # print(f'切分特征: {i}, 阈值:{np.round(Q,2)}, 标志:{S}, 权重误差:{np.round(eca,3)}')
                if eca < minE:  # 找到误差最小的分类方式
                    minE = eca
                    bestClas = predictedVals.copy()
                    bestStump['特征列'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump, minE, bestClas
