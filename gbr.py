from sklearn import ensemble
from sklearn.externals import joblib
gbr = joblib.load("train_model.m")
#户数 写字楼面积 周围网点数 银行 地区
test_X=[[150000,8300000,50,2,3]]
gbr.predict(test_X)

from sklearn import ensemble
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
os.chdir("C:/Users/yufeng/Desktop/python")
df = pd.read_csv('bhan.csv', header=0, encoding='utf-8')
y_train,x_train = df.ix[0:210,0:1],df.ix[0:210,2:]
y_test,x_test = df.ix[210:,0:1],df.ix[210:,2:]
params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params) 
gbr.fit(x_train, y_train)
joblib.dump(gbr, "train_model.m")
y_pre = gbr.predict(x_test)
y_test = np.array(y_test)
m=[]
n=[]
for i,j in enumerate(y_pre):
    print(y_pre[i])
    print(y_test[i][0])
    m.append(y_pre[i]/(y_test[i][0]+y_pre[i])) 
    n.append(y_test[i][0]/(y_test[i][0]+y_pre[i]))
mse = mean_squared_error(n,m)
index=np.arange(1,13,1)
print("MSE: %.4f" % mse)
plt.plot(index,y_pre,'r-',label='predict')
plt.plot(index,y_test,'b-',label='real')
plt.legend(loc='upper right')
plt.show()