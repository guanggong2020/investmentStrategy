import datetime as dt
import time


"""
配置文件
"""


class Arg:
    def __init__(self):
        # 开始日期
        self.start = '20180102'
        # 当前日期，格式2020-01-01
        self.current = dt.datetime.now().strftime('%Y-%m-%d')
        # 当前日期，格式20200101
        self.currentDate = time.strftime("%Y%m%d",time.localtime(time.time()))
        # 沪深300股票数据存储路径
        self.hs300path = '../data/hs300/'
        # 指数数据存储路径
        self.indexpath = '../data/index_data/'

