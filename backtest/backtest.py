# _*_coding:utf-8 _*_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from algo.Adaboost import AdaboostTrainDS, AdaClassify
from util.dataprocess import merge_day_data, split_train_test

plt.rcParams['axes.unicode_minus'] = False  # '-'显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体 黑体


def drawhs300line():
    # 读取沪深300指数数据
    df = pd.read_csv('../data/index_data/399300.csv')
    # print(cal_date.head())
    # df['date'] = df['date'].str.replace('-','')

    data = df[['date', 'yield']][1::20]
    data = data.fillna(0)
    plt.plot(data['date'], data['yield'], label='基准收益率', color='blue', ls=':', lw=2)
    plt.xlabel("时间")
    plt.ylabel("收益率")
    plt.xticks(data['date'][::5], rotation=45)
    plt.axhline(y=0, c='r', ls='--', lw=2)
    plt.legend()
    plt.show()


"""
计算年收益率
"""


def calYearProfit():
    # 起始资金10亿
    capital_base = 1000000000
    # 持仓股票代码
    keepstock = []
    # 持仓股票数量
    keepnum = []
    # 获取上交所交易日期
    cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    # 以20个交易日为时间间隔
    cal_date = cal_date[cal_date.is_open == 1]['cal_date'][1::20]
    # 合并股票数据
    for dt in cal_date:
        df = merge_day_data(str(dt))
        path = '../data/back/' + str(dt) + '.csv'
        df.to_csv(path)
    # 以2018-01-02的数据为训练样本训练得到分类器
    # 获取训练数据集（20190301.csv）
    train_data = pd.read_csv('../data/day_stock_process/20180102.csv')[['open', 'high', 'low', 'close',
                                                                        'pre_close', 'change', 'pct_chg', 'vol',
                                                                        'amount',
                                                                        'label']]

    # 打乱数据集
    num = train_data.shape[0]
    data_index = np.arange(num)
    np.random.shuffle(data_index)
    train_data = train_data.iloc[data_index]
    # 切割数据集获得训练集和测试集的特征矩阵和标签矩阵(训练集：测试集=0.8:0.2)
    train_X, train_Y, test_X, test_Y = split_train_test(train_data)
    # 训练得到弱分类器信息
    weakClass, aggClass = AdaboostTrainDS(train_X, train_Y, maxC=15)
    for dt in cal_date:
        path = '../data/back/' + str(dt) + '.csv'
        df = pd.read_csv(path)[['open', 'high', 'low', 'close',
                                'pre_close', 'change', 'pct_chg', 'vol',
                                'amount',
                                ]]
        # 使用弱分类器对特征矩阵进行分类
        predictions, aggClass = AdaClassify(np.mat(df), weakClass)
        df['prediction'] = predictions
        df['aggClass'] = aggClass
        # 按分类结果降序排序
        df.sort_values(by='prediction', ascending=False, inplace=True)
        # 按分类的累计类别估计值降序排序
        df.sort_values(by='aggClass', ascending=False, inplace=True)
        # 获取前50支股票
        recommend_stock = df.head(50)['ts_code']
        dcode = df['ts_code']
        for kstock in keepstock:
            if kstock not in recommend_stock:
                ...


        for stock in recommend_stock:
            if stock not in keepstock:
                keepstock.append(stock)
                name_index = dcode[dcode.values == stock].index[0]
                price = df['close'][name_index]
                # 买入
                capital_base -= 100000
                # 计算能买多少股
                num = int(100000/price)
                keepnum.append(num)


