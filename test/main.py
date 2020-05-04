#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: Trace
Date: 2020/4/28 18:21
@Desciption: None
"""
from util.data_utils import *

if __name__ == '__main__':

    # 获取当前所有正常上市交易的股票列表
    # get_stock_basic()

    # 根据股票列表下载当前所有正常交易的股票数据，从2018年01月02日开始
    while True:
        download_all_stock()