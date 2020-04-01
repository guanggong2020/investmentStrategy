# _*_coding:utf-8_*_
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd

# 加载数据
df = pd.read_csv('./data/000001.csv', index_col='date')
df = df[['open', 'low', 'high', 'volume', 'close']].copy()
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
# 分割数据
train_x, text_x, train_y, text_y = train_test_split(X, Y, test_size=0.2)
# 使用AdaBoost回归模型
regressor = AdaBoostRegressor()
regressor.fit(train_x, train_y)
pred_y = regressor.predict(text_x)
print("股票收盘价预测结果 ", pred_y)
print("股票收盘价真实价格 ", text_y)

print("均方误差 ", round(mean_squared_error(text_y, pred_y),2))