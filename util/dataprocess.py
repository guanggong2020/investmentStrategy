# _*_coding:utf-8_*_
import math

from sklearn.preprocessing import minmax_scale

from util.data_utils import *

args = Arg()


def mark_stock_yield():
    """
    :return: 标注股票未来20日的收益率,并存入文件中
    """
    pool = pd.read_csv('../data/stock_list/stock_basic.csv')
    for code in pool.ts_code:
        df = pd.read_csv('../data/stock_basic/' + code + '.csv')
        m = df.shape[0]
        df['yield'] = np.zeros((m, 1))
        df['yield'] = np.round((df['close'].shift(-20) - df['close']) / df['close'], 2)
        if not os.path.exists('../data/mark_yield/'):
            os.makedirs('../data/mark_yield/')
        df.to_csv('../data/mark_yield/' + code + '.csv', index=0)


"""
对标注收益率的股票数据按照日期进行合并，去除中间50%的样本数据，合并成一个DataFrame
"""


def merge_day_data(time):
    """
    :param time: 日期
    :return: 按照日期对股票数据进行合并
    """
    # 读取股票基本信息文件，获取股票代码列表
    pool = pd.read_csv('../data/stock_list/stock_basic.csv')
    day_stock = pd.DataFrame()
    for code in pool.ts_code:
        path = '../data/mark_yield/' + code + '.csv'
        df = pd.read_csv(path, index_col='trade_date', parse_dates=True)
        df = df.sort_values('trade_date')
        df = df[['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'yield']]
        data = df[time:time]
        if data is not None:
            day_stock = pd.concat([day_stock, data])
    day_stock = day_stock.sort_values(by='yield', ascending=False)
    l = len(day_stock)
    day_stock = pd.concat([day_stock[:int(0.25 * l)], day_stock[int(0.75 * l):]])
    m = day_stock.shape[0]
    day_stock['label'] = np.ones((m, 1))
    for i in range(m):
        if i > len(day_stock) / 2 - 1:
            day_stock['label'].iloc[i] = -1
    return day_stock


def get_data_by_date():
    """
    :return: 根据上交所交易日历，每隔20个交易日获取已下载的股票数据到指定文件中
    """
    # 获取上交所交易日期
    cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    # 以20个交易日为时间间隔，取当前时间的前十四个时间点
    cal_date = cal_date[cal_date.is_open == 1]['cal_date'][::20]
    for dt in cal_date:
        df = merge_day_data(str(dt))
        path = '../data/day_stock/' + str(dt) + '.csv'
        df.to_csv(path)


def data_preprocessing():
    """
    :return: 数据预处理
    """
    # 获取上交所交易日期
    cal_date = pd.read_csv('../data/trade_cal/trade_cal_sse.csv')
    # 以20个交易日为时间间隔
    cal_date = cal_date[cal_date.is_open == 1]['cal_date'][::20]
    for dt in cal_date:
        path = '../data/day_stock/' + str(dt) + '.csv'
        df = pd.read_csv(path)
        # 删除含有缺失值的数据
        df = df.dropna()
        v_features = ['vol', 'amount']
        for feature in v_features:
            # 取对数
            df[feature] = df[feature].apply(lambda x: math.log(x))
            # 中位数去极值
            df[feature] = filter_extreme_MAD(df[feature], 5)
            # 标准化
            df[feature] = standardize_series(df[feature])
        # price_feature = ['open', 'high', 'low', 'close', 'pre_close']
        # for feature in price_feature:
        #     # 数据归一化
        #     df[feature] = minmax_scale(df[feature])
        df = df[['ts_code','open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'label']]
        df.to_csv('../data/day_stock_process/' + str(dt) + '.csv', index=False)

    """
    构建回测测试集
    """
    def build_test_set():
        cal_date = pd.read_csv("../data/trade_cal/date_test.csv")['cal_date']
        for dt in cal_date:
            df = merge_day_data(str(dt))
            v_features = ['vol', 'amount']
            for feature in v_features:
                # 取对数
                df[feature] = df[feature].apply(lambda x: math.log(x))
                # 中位数去极值
                df[feature] = filter_extreme_MAD(df[feature], 5)
                # 标准化
                df[feature] = standardize_series(df[feature])
            price_feature = ['open', 'high', 'low', 'close', 'pre_close']
            for feature in price_feature:
                # 数据归一化
                df[feature] = minmax_scale(df[feature])
            df = df[
                ['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'label']]
            path = '../data/back/' + str(dt) + '.csv'
            df.to_csv(path)


if __name__ == '__main__':
    # 标注股票收益率
    # mark_stock_yield()
    # 按日期合并股票数据
    # get_data_by_date()
    # 数据预处理
    data_preprocessing()
