# _*_coding:utf-8_*_
import pymongo
from urllib import parse

username = parse.quote_plus("admin")
password = parse.quote_plus("@Admin123")

# 连接mongodb数据库
client = pymongo.MongoClient("mongodb://{0}:{1}@134.175.192.53:27017/".format(username,password))

# 指定数据库
db = client.bishe

# 指定集合
collection = db.gupiao_code

# 查询
info = collection.find_one()
print('select:', info['code'])