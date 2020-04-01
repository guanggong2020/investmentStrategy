from util.data_utils import getCurrentTime
import datetime as dt


# -------------------参数配置----------------- #


class Arg:
    def __init__(self):
        # 开始日期
        self.start = '1980-01-01'
        # 当前日期
        self.current = dt.datetime.now().strftime('%Y-%m-%d')
        # 沪深300股票数据存储路径
        self.hs300path = '../data/hs300/'
        # 指数数据存储路径
        self.indexpath = '../data/index_data/'
