# _*_coding:utf-8 _*_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from algo.Adaboost import AdaboostTrainDS, AdaClassify
from util.dataprocess import merge_day_data, split_train_test

plt.rcParams['axes.unicode_minus'] = False  # '-'显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体 黑体

"""
绘制沪深300指数收益率以及策略收益率的比较图
"""


def drawprofitline():
    # 读取沪深300指数数据
    # df = pd.read_csv('../data/index_data/399300.csv')
    # df['date'] = df['date'].str.replace('-','')
    df = calhs300yearprofit()
    stockprofit = pd.read_csv('../data/back/result.csv')
    plt.plot(df['date'], df['yield'], label='基准收益率', color='blue', ls=':', lw=2)
    plt.plot(df['date'], stockprofit['yield'], label='策略收益率', color='black', ls=':', lw=2)
    plt.xlabel("时间")
    plt.ylabel("收益率")
    plt.xticks(df['date'][::5], rotation=45)
    plt.axhline(y=0, c='r', ls='--', lw=2)
    plt.legend()
    plt.savefig('../image/profit_v1.png')
    plt.show()


"""
计算年收益率
"""


def calStockYearProfit():
    # 起始资金1000万
    capital_base = 10000000
    # 记录买卖后的资金
    capital_change = capital_base
    # 持仓股票代码和每只股票熟练
    keepstock = DataFrame({'code': [], 'number': [], 'price': []})
    # 存储日期和收益率
    save_date_yield = DataFrame({'date': [], 'yield': []})
    # 读取交易日期
    cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    # 以20个交易日为时间间隔
    cal_date = cal_date[cal_date.is_open == 1]['cal_date'][1::20]
    # 以2018-01-02的数据为训练样本训练得到分类器
    # 获取训练数据集（20180102.csv）
    train_data = pd.read_csv('../data/back/20180102.csv')[['open', 'high', 'low', 'close',
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
        df = pd.read_csv(path)[['ts_code', 'open', 'high', 'low', 'close',
                                'pre_close', 'change', 'pct_chg', 'vol',
                                'amount',
                                ]]
        # 测试集特征样本
        test_X = np.mat(df.iloc[:, 1:].values)
        # 使用弱分类器对特征矩阵进行分类
        predictions, aggClass = AdaClassify(test_X, weakClass)
        df['prediction'] = predictions
        df['aggClass'] = aggClass
        # 按分类结果降序排序
        df.sort_values(by='prediction', ascending=False, inplace=True)
        # 按分类的累计类别估计值降序排序
        df.sort_values(by='aggClass', ascending=False, inplace=True)
        df = df.dropna()
        # 获取前50支股票
        recommend_stock = df.head(50)['ts_code']
        # print(df.head(5)[['ts_code','open']])
        dcode = df['ts_code']
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
                        price = df['close'][name_index]
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
                price = df['open'][name_index]
                # 买入
                capital_change -= 100000
                # 计算能买多少股
                num = int(100000 / price)
                i += 1
                keepstock = keepstock.append({'code': stock, 'number': num, 'price': price}, ignore_index=True)
        else:
            for stock in recommend_stock:
                name_index = dcode[dcode == stock].index[0]
                price = df['open'][name_index]
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
                    price = df['close'][name_index]
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
        save_date_yield = save_date_yield.append({'date': str(dt), 'yield': rate}, ignore_index=True)
    return save_date_yield


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
    # df = calhs300yearprofit()
    # print(df)
    drawprofitline()