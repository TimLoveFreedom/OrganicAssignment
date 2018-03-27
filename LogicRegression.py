
# coding: utf-8

# 1.导入需要用到的常用库

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[ ]:

2.加载数据集


# In[11]:

#load data
data=pd.read_csv('organics.csv')
# 从数据集中移除 'ORGYN' 这个特征，并将它存储在一个新的变量中。
labels = data['ORGYN']
features = data.drop('ORGYN', axis = 1)


# In[12]:

features.head()


# In[ ]:

3.数据预处理
    清洗数据的4C准则：
    Correcting: 处理异常值
    Completing: 处理缺失值
    对于定性数据，一般用众数替代缺失值
    对于定量数据，一般用均值、中位数或均值+标准差替代缺失值
    删除不影响分析挖掘的特征
    Creating: 特征工程。创建新的特征，挖掘隐藏的信息。
    Converting: 转换数据格式。例如，将分类型数据进行编码，便于数学计算。


# In[14]:

# Fill empty and NaNs values with NaN
features = features.fillna(np.nan)
# Check for Null values
features.isnull().sum()


# 4.模型训练
#     

# In[ ]:

#数据切分
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)  #cv默认为3
clf.fit(X_train, y_train)  # default(X_train  y_train) or optimal(X_train['agegroup'])


# In[ ]:

#输出最优的模型参数
print(clf.best_params_)


# In[ ]:

#选择出特征权值比较大的feature


# In[ ]:

#rfe来筛选特征
estimator=LogisticRegression()
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X_train, y_train)


# In[ ]:

#模型预测比较（all  inputs  or  optimal inputs  可以使用准确率比较哪个效果更好）
#应该是optimal  inputs效果比较好，因为all inputs 噪音比较大



# In[ ]:



