#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


#data1 = pd.read_csv('/home/yash/work/ComputerVision/train.csv')
#data = data1.head(1000)
#data.to_pickle('/home/yash/work/ComputerVision/train.pkl')

data = pd.read_csv('/home/yash/work/ComputerVision/train.csv')
X = data.drop(columns=['label'])
y = data['label']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(train_X, train_y)

#predictions = model.predict(test_X)
#predictions

#score = accuracy_score(test_y, test_y)
#score

tree.export_graphviz(model, out_file='/home/yash/work/ComputerVision/DT-model.dot',
                    feature_names=X.columns,
                    class_names=sorted(str(train_y.unique())),
                    label='all',
                    rounded=True,
                    filled=True)


# In[27]:


data = pd.read_csv('/home/yash/work/ComputerVision/train.csv')

data.columns


# In[ ]:




