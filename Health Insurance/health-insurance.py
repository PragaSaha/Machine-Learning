#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[25]:


# Building a model to predict whether a customer would be interested in Health Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.


# In[3]:


train_data= pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test_data = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')


# In[27]:


train_data.info()


# 1. 12 Columns and 381109 Row.
# 2. Out of which one is dependent variable and rest 11 are independent variables.
# 3. Columns of Type and count: Object - 3, Float-3 , int - 6 
# 4. No variable column has null/missing values.

# In[28]:


train_data.describe().T


# 1. Mean value is less than median value of column which is represented by 50%(50th percentile) in index column- Driving_License,Region_Code, Annual_Premium, Policy_Sales_Channel
# 2. Mean value is more than median value of column: Age, Vintage, Previously_Insured
# 3. There is notably a large difference between 75th %tile and max values of predictors: Annual_Premium, Vintage
# 
# 4. Observation from above, Outliers present.

# 

# In[4]:


train_data.Response.value_counts()


# In[5]:


sns.pairplot(train_data, hue ='Response')


# This tells us vote count of each Response in descending order.
# “Response” has values concentrated in the categories 0 and 1.

# In[30]:


train_data.corr()


# In[31]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(train_data.corr(), annot=True)


# 1. Very Low Positive Correlation and Negative Correlation are observed in range -0.58 to 0.22
# 2. Since correlation is approximately to zero we can infer there is no linear relationship between these two predictors.However it is safe to drop these features in case you’re applying Linear Regression model to the dataset.

# In[32]:


# Box Plot for Outliers
for columns in train_data:
    if train_data[columns].dtypes == object:
        print(columns)
        
    else:    
        plt.figure()
        train_data.boxplot([columns])


# In our data set only “Annual_Premium” feature column shows outliers.

# In[33]:


# check the linearity of the variables - distribution graphs and look for skewness of features
# - UNIVARIANT ANALYSIS
train_data.hist(bins= 20, figsize=(20,15))
plt.show()


# In[34]:


# BIVARIANT ANALYSIS
from pandas.plotting import scatter_matrix
# pair analysis
scatter_matrix(train_data, alpha=0.2, figsize=(60, 60))
plt.show()


# In[35]:


outliers = []
def detect_outliers_zscore(data):
    thres = 3 #optimal
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers# Driver code
sample_outliers = detect_outliers_zscore(train_data.Annual_Premium)

print(len(sample_outliers))


# In[36]:


# Handling Outliers - As the mean value is highly influenced by the outliers, it is advised to replace the outliers with the median value.
median = np.median(train_data.Annual_Premium)# Replace with median
for i in sample_outliers:
    c = np.where(train_data.Annual_Premium==i, 14, train_data.Annual_Premium)
print("Sample: ", train_data.Annual_Premium)
print("New array: ",c)
# print(x.dtype)


# In[37]:


# Split the Data
from sklearn.model_selection import train_test_split

X=train_data.drop(['id','Response'],axis=1)
y=train_data.Response


# In[38]:


# Categorical column - Label Encoding
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le= LabelEncoder()
for feature in X.columns: # Loop through all columns in the dataframe
    if X[feature].dtype == 'object': # Only apply for columns with categorical strings
        X[feature]= le.fit_transform(X[feature])      

X.head()


# In[39]:


# Labelling Test Data
for feature in test_data.columns: # Loop through all columns in the dataframe
    if test_data[feature].dtype == 'object': # Only apply for columns with categorical strings
        test_data[feature]= le.fit_transform(test_data[feature])


# In[40]:


# Standard scaler helps us to make all variable in same unit.
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()

std_x = standard.fit_transform(X)


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 7)# Stratify y to keep the class proportions consistent
# Print the number of train and test records.
print("Train Data:",X_train.shape)
print("Test Data:",X_test.shape)


# In[42]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_nb = gnb.fit(X_train, y_train).predict(X_test)
y_pred_nb


# In[43]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_nb)


# In[44]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_nb))


# In[45]:



pred_test = gnb.predict(test_data.drop('id',axis=1))
submission = pd.DataFrame({'id':np.arange(381110, 381110+len(pred_test)),'Response':pred_test})

submission.to_csv('.//sample_submission.csv',index=False)


# In[46]:


submission.head(10)

