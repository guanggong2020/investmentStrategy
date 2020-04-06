# _*_coding:utf-8_*_
import pandas as pd
import numpy as np
from config import Arg
from util.data_utils import *

if __name__ == '__main__':
    args = Arg()
    """
    下载沪深300股票数据
        1、下载沪深300股票信息
        2、批量下载沪深300股票数据
    """

    # get_hs300_code_name(args.hs300path)
    # download_hs300_stock_data(args.start,args.current,args.hs300path)

    """
    下载沪深300指数数据
    """
    # get_hs300_data(args.start,args.current,args.indexpath)

    """
    下载交易所交易日历数据
    """
    # trade_cal_sse('20180101')

    """
    查询当前所有正常上市交易的股票列表
    """
    # get_stock_basic()

    """
    下载所有上市的股票数据 时间20180101-20200405
    """
    # download_all_stock()

