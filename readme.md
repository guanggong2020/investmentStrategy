# 前期准备
## 数据分析的基本流程
1. **商业理解**：数据挖掘不是我们的目的，我们的目的是更好地帮助业务，所以第一步我们要从商业的角度理解项目需求，在这个基础上，再对数据挖掘的目标进行定义。
2. **数据理解**：收集部分数据，然后对数据进行探索，包括数据描述、数据质量验证等。这有助于你对收集的数据有个初步的认知。
3. **数据准备**：开始收集数据，并对数据进行清洗、数据集成等操作，完成数据挖掘前的准备工作。
4. **模型建立**：选择和应用各种数据挖掘模型，并进行优化，以便得到更好的分类结果。
5. **模型评估**：对模型进行评价，并检查构建模型的每个步骤，确认模型是否实现了预定的商业目标。
6. **上线发布**：模型的作用是从数据中找到金矿，也就是我们所说的“知识”，获得的知识需要转化成用户可以使用的方式，呈现的形式可以是一份报告，也可以是实现一个比较复杂的、可重复的数据挖掘过程。数据挖掘结果如果是日常运营的一部分，那么后续的监控和维护就会变得重要。
## 开发测试环境
JuPyter notebook + Python3.7
<br/>
使用的库：NumPy、Pandas、matplotlib,sklearn
## 前期学习
### Python基础语法
#### 输入输出

```
str = input('...')
print(str)
```
#### if...else...

```
if a > 0:
    ...
else:
    ...
```
#### 循环语句：for … in

```
for i in range(n):
    ...
```
#### 循环语句: while

```
while a < 100:
    ...
```
#### 列表：[]

```
lists = ['a','b','c']
lists.append('d')   #尾部追加
print len(lists)    #列表长度
lists.insert(0,'mm')#在列表的指定位置插入元素
lists.pop()         #删除列表尾部的元素
```
#### 元组（tuple）

```
tuples = ('A','B')
print(tuples[0])    #tuple一旦初始化就不能修改
```
#### 字典 {dictionary}

```
score = {'chinese':90,'math':91}
score['python'] = 92    #添加元素
score.pop('chinese')    #删除元素
print('math' in score)  #查看math在score中是否存在
print(score.get('math') #获取math对应的值
print(score.get('chinese',90)#获取chinese对应的值如果获取不到就使用默认值90
```
#### 集合：set

```
s = set(['a', 'b', 'c'])
s.add('d')
s.remove('b')
print('c' in s)
```
#### 函数

```
def methodName(param):
    ...
```
### 使用Numpy快速处理数据
#### 简介
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NumPy是Python中使用最多的第三方库，而且还是SciPy、Pandas等数据科学的基础库，NumPy所提供的数据结构是Python数据分析的基础。
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与Python数据结构中的列表list相比，NumPy提供的数组结构更节省内存和计算资源。Python的列表除了需要存储对应的元素，还需要存储对应的指针，且list的元素在系统内存中是分散存储的，而NumPy数组存储在一个均匀连续的内存块中。这样数组计算遍历所有的元素，不像列表list还需要对内存地址进行查找，从而节省了计算资源。
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NumPy里有两个重要的对象：ndarray解决了多维数组问题，而ufunc则是解决对数组进行处理的函数。
#### ndarray对象
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ndarray实际上是多维数组的含义。在NumPy数组中，维数称为秩，一维数组的秩为1，二维数组的秩为2，以此类推。在NumPy中，每一个线性的数组称为一个轴（axes），其实秩就是描述轴的数量。
#### 创建数组

```
import numpy as np
a = np.array([1,2,3])   #创建一维数组
b = np.array([[1,2,3],[4,5,6],[7,8,9]]) #创建多维数组
print(b.shape)  #获取数组大小
print(a.dtype)  #获取元素属性
```
#### 结构数组

```
persontype = np.dtype({'names':['name','age','chinese','math','english'],'formats':['S32','i','i','i','f']})#定义结构类型
peoples = np.array([("ZhangFei",32,75,100,90),("GuanYu",24,85,96,88.5),("ZhaoYun",28,85,92,96.5),("HuangZhong",29,65,85,100)],dtype=persontype)
ages = peoples[:]['age']#获取age的值
print(np.mean(ages))#计算平均值
```
#### ufunc运算
#### 连续数组的创建

```
#两者的结果是一样的，都是创建等差数组，结果都是[1,3,5,7,9]
x1 = np.arange(1,11,2)  #初始值、终值、步长（不包含终值）
x2 = np.linspace(1,9,5)#初始值、终值、元素个数（包含终值）
```
#### 算数运算

```
print(np.add(x1, x2))           #加
print(np.subtract(x1, x2))      #减
print(np.multiply(x1, x2))      #乘
print(np.divide(x1, x2))        #除
print(np.power(x1, x2))         #指数
print(np.remainder(x1,x2))      #取余
```
#### 统计函数
#### 计算数组/矩阵的最大值和最小值

```
import numpy as np
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(np.amin(a))   #最小值
print(np.amin(a,0)) #延着axis=0轴的最小值[1,2,3]
print(np.amin(a,1)) #延着axis=1轴的最小值[1,4,7]
print(np.amax(a))   #最大值
print(np.amax(a,0)) #延着axis=0轴的最大值[7,8,9]
print(np.amax(a,1)) #延着axis=1轴的最大值[3,6,9]
```
#### 统计最大值与最小值之差 ptp()

```
c = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.ptp(c))    #数组中最大值与最小值的差
print(np.ptp(c,0))  #沿着axis=0轴的最大值与最小值之差[6,6,6]
print(np.ptp(c,1))  #沿着axis=1轴的最大值与最小值之差[2,2,2]
```
#### 统计数组的百分位数 percentile()

```
#p = 0:求最小值 p = 50:求平均值 p=100:求最大值
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
print np.percentile(a, 50)          #5.0
print np.percentile(a, 50, axis=0)  #[4. 5. 6.]
print np.percentile(a, 50, axis=1)  #[2. 5. 8.]
```
#### 统计数组中的中位数median()、平均数mean()

```
c = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.median(c))             #5.0
print(np.median(c,0))           #[4. 5. 6.]
print(np.median(c,1))           #[2. 5. 8.]
print(np.mean(c))               #5.0
print(np.mean(c,0))             #[4. 5. 6.]
print(np.mean(c,1))             #[2. 5. 8.]
```
#### 统计数组中的加权平均值average()

```
a = np.array([1,2,3,4])
wts = np.array([1,2,3,4])
print(np.average(a))    #求加权平均数，默认每个元素的权重是相同的
print(np.average(a,weights=wts))    #(1*1+2*2+3*3+4*4)/(1+2+3+4)=3.0
```
#### 统计数组中的标准差std()、方差var()

```
a = np.array([1,2,3,4])
print(np.std(a))
print(np.var(a))
```
#### NumPy排序

```
#sort(a, axis=-1, kind=‘quicksort’, order=None)
a = np.array([[4,3,2],[2,4,1]])
print(np.sort(a))           #[[2 3 4][1 2 4]]
print(np.sort(a,axis=None)) #[1 2 2 3 4 4]
print(np.sort(a,axis=0))    #[[2 3 1][4 4 2]]
print(np.sort(a,axis=1))    #[[2 3 4][1 2 4]]
```
kind可以指定quicksort、mergesort、heapsort分别表示快速排序、合并排序、堆排序<br/>
order字段，对于结构化的数组可以指定按照某个字段进行排序。
<br/>
<br/>
### Pandas
Pandas提供的基础数据结构DataFrame与json的契合度很高，转换起来就很方便,且基于Series和 DataFrame这两个核心数据结构，Pandas可以对数据进行导入、清洗、处理、统计和输出。
#### Series
Series是个定长的字典序列。说是定长是因为在存储的时候，相当于两个ndarray。Series有两个基本属性：index 和 values。在Series结构中，index默认是0,1,2,……递增的整数序列，当然我们也可以自己来指定索引，比如index=[‘a’, ‘b’, ‘c’, ‘d’]。

```
import pandas as pd
from pandas import Series
x1 = Series([1,2,3,4])
x2 = Series(data=[1,2,3,4], index=['a', 'b', 'c', 'd'])
d = {'a':1, 'b':2, 'c':3, 'd':4}
x3 = Series(d)

output:
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        a    1
        b    2
        c    3
        d    4
        dtype: int64
```
#### DataFrame

```
import pandas as pd
from pandas import DataFrame
data = {'Chinese':[66,95,93,90,80],'Math':[30,98,96,77,90],'English':[65,85,92,88,90]}
df1 = DataFrame(data)
print(df1)

output:
   Chinese  Math  English
0       66    30       65
1       95    98       85
2       93    96       92
3       90    77       88
4       80    90       90


```

```
df2 = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'], columns=['English', 'Math', 'Chinese'])
              Chinese  Math  English
ZhangFei         66    30       65
GuanYu           95    98       85
HuangZhong       93    96       92
DianWei          90    77       88
LvBu             80    90       90
```
#### 数据的导入和输出
Pandas允许直接从xlsx，csv等文件中导入数据，也可以输出到xlsx, csv等文件

```
score = DataFrame(pd.read_excel('data.xlsx'))
score.to_excel('data1.xlsx')
```
#### 数据清洗
#### 删除 DataFrame 中的不必要的列或行

```
df2 = df2.drop(columns=['Chinese']) #删除列
df2 = df2.drop(index=['ZhangFei'])  #删除行
```
#### 重命名列名columns，让列表名更容易识别
对DataFrame中的columns进行重命名，可以直接使用rename(columns=new_names, inplace=True) 函数

```
df2.rename(columns={'Chinese': 'YuWen', 'English': 'Yingyu'}, inplace = True)
```
#### 去除重复的值

```
df = df.drop_duplicates() #去除重复行
```
#### 更改数据格式

```
df2['Chinese'].astype('str')
```
#### 数据间的空格

```
df2['Chinese']=df2['Chinese'].map(str.strip) 
df2['Chinese']=df2['Chinese'].map(str.lstrip)#删除左边空格
df2['Chinese']=df2['Chinese'].map(str.rstrip)#删除右边空格
df2['Chinese']=df2['Chinese'].str.strip('$')#删除数据里特殊的符号
```
#### 大小写转换

```
df2.columns = df2.columns.str.upper()#全部大写
df2.columns = df2.columns.str.lower()#全部小写
df2.columns = df2.columns.str.title()#首字母大写
```
#### 查找空值

```
df.isnull()#查看DataFrame数据表中哪个地方存在空值
df.isnull().any()#哪列存在空值
```
#### 使用apply函数对数据进行清洗

```
df['name'] = df['name'].apply(str.upper)#将name列的数值都进行大小写转换
```

```
#定义double_df函数是将原来的数值*2进行返回，然后对df1中的“语文”列的数值进行*2处理
def double_df(x):
           return 2*x
df1[u'语文'] = df1[u'语文'].apply(double_df)
```

```
#在df中新增两列
def plus(df,n,m):
    df['new1'] = (df[u'语文']+df[u'英语']) * m
    df['new2'] = (df[u'语文']+df[u'英语']) * n
    return df
df1 = df1.apply(plus,axis=1,args=(2,3,))#args是传递的两个参数，即n=2, m=3
```

#### 统计函数

```
count():统计个数，空值和NaN不统计
min():最小值
max():最大值
sum():总和
mean():平均数
median():中位数
var():方差
std():标准差
argmin():统计最小值的索引位置
argmax():统计最大值的索引位置
describe():一次性输出多个统计指标count、mean、std、min、max
```
#### 数据表合并

```
df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1':range(5)})
df2 = DataFrame({'name':['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2':range(5)})
df3 = pd.merge(df1, df2, on='name')#基于指定列进行连接
df3 = pd.merge(df1, df2, how='inner')#inner内连接
df3 = pd.merge(df1, df2, how='left')#左连接是以第一个DataFrame为主进行的连接，第二个DataFrame作为补充。
df3 = pd.merge(df1, df2, how='right')#右连接是以第二个DataFrame为主进行的连接，第一个DataFrame作为补充。
df3 = pd.merge(df1, df2, how='outer')#外连接相当于求两个DataFrame的并集。
```
### 数据集获取
  Tushare实现了对股票，基金等金融数据从数据采集、清洗加工到数据存储的过程，且返回的绝大部分的数据格式都是pandas的DataFrame类型，非常便于用pandas/NumPy/Matplotlib进行数据分析和可视化<br/>
####   可以获取的交易数据主要有
1. 历史行情数据
1. 复权历史数据
1. 实时行情数据
1. 历史分笔数据
1. 实时报价数据
1. 当日历史分笔
1. 大盘指数列表
1. 大单交易数据

```
import tushare as ts
ts.get_hist_data('600848') #一次性获取全部日k线数据
```
### 算法
#### AdaBoost
#### 原理
先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练集样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多的关注，然后基于调整后的样本分布来训练下一个基学习器。如此重复执行，直至基学习器数目达到事先指定的值T,最终将这T个基学习器进行加权结合

#### 过程
#### 1. 计算样本权重
训练数据中的每个样本，赋予其权重，即样本权重，用向量D表示，这些权重都初始化成相等值。假设有n个样本的训练集：<br/>
<br/>
![image](https://cuijiahua.com/wp-content/uploads/2017/11/mL_10_3.png)<br/>
<br/>
设定每个样本的权重都是相等的，即1/n。
#### 2、计算错误率
利用第一个弱学习算法h1对其进行学习，学习完成后进行错误率ε的统计：<br/>
![image](https://cuijiahua.com/wp-content/uploads/2017/11/mL_10_4.png)
#### 3、计算弱学习算法权重
弱学习算法也有一个权重，用向量α表示，利用错误率计算权重α：<br/>
![image](https://www.cuijiahua.com/wp-content/uploads/2017/11/mL_10_5.png)
#### 4、更新样本权重
在第一次学习完成后，需要重新调整样本的权重，以使得在第一分类中被错分的样本的权重，在接下来的学习中可以重点对其进行学习：<br/>
<br/>
![image](https://cuijiahua.com/wp-content/uploads/2017/11/mL_10_6.png)<br/>
<br/>
其中，ht(xi) = yi表示对第i个样本训练正确，不等于则表示分类错误。Zt是一个归一化因子：<br/>
<br/>
![image](https://www.cuijiahua.com/wp-content/uploads/2017/11/mL_10_7.png)
<br/>
<br/>
将两个公式进行合并，化简如下：<br/>
<br/>
![image](https://cuijiahua.com/wp-content/uploads/2017/11/mL_10_8.png)
<br/>
<br/>
#### 5、AdaBoost算法
重复进行学习，这样经过t轮的学习后，就会得到t个弱学习算法、权重、弱分类器的输出以及最终的AdaBoost算法的输出
<br/><br/>
sign(x)是符号函数。具体过程如下所示：<br/><br/>
![image](https://cuijiahua.com/wp-content/uploads/2017/11/mL_10_10.png)

#### AdaBoost算法总结如下：<br/><br/>
![image](https://cuijiahua.com/wp-content/uploads/2017/11/mL_10_9.png)
### 基于单层决策树构建弱分类器
单层决策树(decisionstump)是一种简单的决策树，它仅仅基于单个特征来做决策。由于这棵树实际上只有一次分裂的过程，因此它实际上就是一个树桩。
#### 构建简单数据集

```
import pandas as pd
import numpy as np
#获得特征矩阵和标签矩阵
def get_Mat(path):
    dataSet = pd.read_table(path,header = None)
    xMat = np.mat(dataSet.iloc[:,:-1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    return xMat,yMat
    xMat,yMat = get_Mat('simpdata.txt')
```
#### 构建单层决策树
建立两个函数实现单层决策树:<br/>
第一个函数用来测试是否有某个值小于或者大于我们正在预测的阙值<br>
第二个函数会在一个加权数据集中循环，并找到具有最低错误率的单层决策树<br>

```
伪代码：
将最小错误率minE设为正无穷
对数据集中的每一个特征（第一层循环）：
    对每个步长（第二层循环）：
        对每个不等号（第三次循环）：
            建立一棵单层决策树并利用加权数据集对它进行预测
            如果错误率低于minE，则将当前决策树设为最佳单层决策树
返回最佳单层决策树
```


```
"""
函数功能：单层决策树分类函数
参数说明:
xMat: 数据矩阵
i: 第i列，也就是第几个特征
Q: 阈值
S: 标志
返回:
re: 分类结果
"""
def Classify0(xMat,i,Q,S):
    re = np.ones((xMat.shape[0],1)) #初始化re为1
    if S == 'lt':
        re[xMat[:,i] <= Q] = -1 #如果小于阈值,则赋值为-1
    else:
        re[xMat[:,i] > Q] = -1 #如果大于阈值,则赋值为-1
    return re
```

```
"""
函数功能：找到数据集上最佳的单层决策树
参数说明:
    xMat：特征矩阵
    yMat：标签矩阵
    D：样本权重
返回:
    bestStump：最佳单层决策树信息
    minE：最小误差
    bestClas：最佳的分类结果
"""
def get_Stump(xMat,yMat,D):
    m,n = xMat.shape #m为样本个数，n为特征数
    Steps = 10 #初始化一个步数
    bestStump = {} #用字典形式来储存树桩信息
    bestClas = np.mat(np.zeros((m,1))) #初始化分类结果为1
    minE = np.inf #最小误差初始化为正无穷大
    for i in range(n): #遍历所有特征
        Min = xMat[:,i].min() #找到特征中最小值
        Max = xMat[:,i].max() #找到特征中最大值
        stepSize = (Max - Min) / Steps #计算步长
        for j in range(-1, int(Steps)+1):
            for S in ['lt', 'gt']: #大于和小于的情况，均遍历。lt:less than，gt:greater than
                Q = (Min + j * stepSize) #计算阈值
                re = Classify0(xMat, i, Q, S) #计算分类结果
                err = np.mat(np.ones((m,1))) #初始化误差矩阵
                err[re == yMat] = 0 #分类正确的,赋值为0
                eca = D.T * err #计算误差
                #print(f'切分特征: {i}, 阈值:{np.round(Q,2)}, 标志:{S}, 权重误差:{np.round(eca,3)}')
                if eca < minE: #找到误差最小的分类方式
                    minE = eca
                    bestClas = re.copy()
                    bestStump['特征列'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump,minE,bestClas
```
