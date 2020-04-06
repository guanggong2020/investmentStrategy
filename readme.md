# 文件说明
```
algo(算法实现)：
    stump.py 单层决策树分类实现以及最佳单层决策树的获取
    Adaboost.py Adaboost训练获得弱分类器数组以及集成弱分类器成强分类器
data（存储数据）:
    day_stock:存储指定时间点所有的股票数据
    day_stock_process:存储添加了收益率和标签的指定时间点所有的股票数据
    hs300：沪深300股票数据
    index_data:指数数据
    mark_yield:标注股票未来20天收益率的全部股票数据
    stock_basic:存储当前上市的所有股票数据
    trade_cal:交易所交易日历
image:图片
learning:学习记录过程
test:测试
    adaboost_prediction.py:多股票预测，目前正在进行的工作
    adaboost_price.py:仿造预测房价预测股票价格，目前来看这种做法不行
    SingleStockTest.py:单股票预测
    test:测试代码文件，忽略
util:工具
    data_util.py:一些下载，更新股票数据的函数
    loadData.py:下载股票数据
    dataprocess:股票预处理函数
config.py:配置信息，存储一些常量
draw.py：画图
```
