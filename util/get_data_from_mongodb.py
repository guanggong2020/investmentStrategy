# _*_coding:utf-8_*_

from urllib import parse
import pymongo
import numpy as np

from pandas import DataFrame

# 账号
username = parse.quote_plus("admin")
# 密码
password = parse.quote_plus("@Admin123")

# 连接mongodb数据库
client = pymongo.MongoClient("mongodb://{0}:{1}@134.175.192.53:27017/".format(username, password))

# 指定数据库
db = client.bishe

# 指定股票代码集合
code_collection = db.gupiao_code

# 指定股票数据集合
data_collection = db.gupiao_data

# 查询单条数据
# info = code_collection.find_one()
# print('select:', info['code'])


""""
从mongodb中获取A股所有的股票数据
"""
# 获取股票列表
code_list = code_collection.find({}, {"_id": 0, "code": 1},no_cursor_timeout=True)
for x in code_list:
    # 根据股票代码获取股票数据
    query = {"code": x['code']}
    stock_data = data_collection.find(query,no_cursor_timeout=True)

    # 使用DataFrame进行存储
    df = DataFrame(
        {'date': [], 'code': [], 'name': [], 'closingPrice': [], 'maxPrice': [], 'minPrice': [], 'openingPrice': [],
         'previousClose': [], 'change': [], 'quoteChange': [], 'turnoverRate': [], 'volume': [], 'turnover': [],
         'totalMarketCapitalization': [], 'marketCapitalization': []})

    for stock in stock_data:
        df = df.append(
            {'date': stock['date'], 'code': stock['code'], 'name': stock['name'], 'closingPrice': stock['closingPrice'],
             'maxPrice': stock['maxPrice'], 'minPrice': stock['minPrice'], 'openingPrice': stock['openingPrice'],
             'previousClose': stock['previousClose'], 'change': stock['change'], 'quoteChange': stock['quoteChange'],
             'turnoverRate': stock['turnoverRate'], 'volume': stock['volume'], 'turnover': stock['turnover'],
             'totalMarketCapitalization': stock['totalMarketCapitalization'],
             'marketCapitalization': stock['marketCapitalization']}, ignore_index=True)
    stock_data.close()
    # 去重
    df = df.drop_duplicates()
    # 标注未来20天的股票收益率
    m = df.shape[0]
    df['yield'] = np.zeros((m, 1))
    df['yield'] = np.round((df['closingPrice'].shift(20) - df['closingPrice']) / df['closingPrice'], 2)
    df.to_csv('../data/data_from_mongodb/' + x['code'] + '.csv')
code_list.close()