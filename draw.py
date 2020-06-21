# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import numpy as np
from matplotlib.font_manager import FontProperties

from backtest.back import calhs300yearprofit

plt.rcParams['axes.unicode_minus'] = False  # '-'显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体 黑体


# ------------------------绘制单支股票折线图------------------------ #
def drawStockTrend(name, start, imageName):
    # name:公司名字  name:开始时间 imageName图片保存名称 绘制从指定开始时间到当前时间的股票的开盘价、最低价、最高价、收盘价、五日均线和十日均线走势
    # 根据公司名字从hs300获取股票代码code
    df = pd.read_csv('./data/hs300/hs300.csv')
    dname = df['name']
    dcode = df['code']
    code_index = dname[dname.values == name].index[0]
    target_code = dcode[code_index]
    # 获取股票数据[start:now]
    data = pd.read_csv('./data/hs300/' + str(target_code) + '.csv', index_col='date', parse_dates=['date'])
    now = dt.datetime.now().strftime('%Y-%m-%d')
    data = data[start:now]
    # 创建画布
    plt.figure(figsize=(20, 8), dpi=100)
    plt.title(name)
    plt.xticks(rotation=30)
    plt.plot(data.index, data['open'], label='open', marker='o', linestyle=':', linewidth=1, markersize=3, color='gray')
    plt.plot(data.index, data['high'], label='high', marker='o', linestyle=':', linewidth=1, markersize=3,
             color='green')
    plt.plot(data.index, data['low'], label='low', marker='o', linestyle=':', linewidth=1, markersize=3, color='blue')
    plt.plot(data.index, data['close'], label='close', marker='o', linestyle='-', linewidth=2, markersize=4,
             color='red')

    for x, y in zip(data.index, data['close']):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom', color='red')
    # 显示图例
    plt.legend()

    plt.xlabel("时间")
    plt.ylabel("价格")
    # 修改刻度
    plt.xticks(data.index[::1])

    # 添加网格显示
    plt.grid(True, linestyle="--", alpha=1)

    # 保存图片
    if not os.path.exists('./image/'):
        os.makedirs('./image/')
    plt.savefig('./image/' + imageName)

    plt.show(block=True)
    plt.close()


# drawStockTrend(u'浦发银行', '2020-02-19', u'浦发银行.png')
# 绘制弱分类器个数与准确率折线图
def draw_accuracy():
    x = range(200)
    df = pd.read_csv('./data/accuracy/accuracy_v6.csv')
    y_train_accuracy = df['训练准确率']
    y_test_accuracy = df['测试准确率']
    y_cross_validation = df['5折交叉验证准确率']
    plt.plot(x, y_train_accuracy, ls=':', lw=2, label='训练准确率')
    plt.plot(x, y_test_accuracy, ls=':', lw=2, label='测试准确率')
    plt.plot(x, y_cross_validation, ls=':', lw=2, label='5折交叉验证准确率', color="red")
    plt.legend()
    plt.savefig('./image/accuracy_v6.png')
    plt.show()





