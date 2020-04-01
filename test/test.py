from pandas import DataFrame
data = {'code':[],'accuracy':[]}
new = DataFrame({'code':['000001'],'accuracy':[0.98]})
df1 = DataFrame(data)
df1 = df1.append(new,ignore_index=True)
new = DataFrame({'code':['000002'],'accuracy':[0.98]})
df1 = df1.append(new,ignore_index=True)
print(df1)