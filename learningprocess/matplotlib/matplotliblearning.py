# _*_coding:utf-8 _*_
import matplotlib.pyplot as plt
import numpy as np
import time

filename = time.strftime('%H-%M-%S', time.localtime(time.time()))
x = np.linspace(0.5, 3.5, 100)  # 在0.5至3.5之间均匀取100个数
y = np.sin(x)
y1 = np.random.randn(100)  # 在标准正太分布中随机地取100个数

"""
绘制折线图
"""


def pltLine(saveImg=False):
    plt.plot(x, y, ls='-', lw=2, label='plot figure')  # ls:线条风格 lw:线条宽度 label：标记图像内容的标签文本，需plt.legend()
    if saveImg:
        plt.savefig('./image/' + filename + '.png')
    plt.legend()
    plt.show()


"""
绘制散点图
"""


def pltscatter(saveImg=False):
    x = np.linspace(0.05,10,1000)
    y = np.random.rand(1000)
    plt.scatter(x, y, c="g", label='scatter figure')
    plt.legend()
    if saveImg:
        plt.savefig('./image/' + filename + '.png')
    plt.show()


if __name__ == '__main__':
    # pltLine(True)
    pltscatter(True)
