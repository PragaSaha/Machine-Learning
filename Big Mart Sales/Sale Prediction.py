#!/usr/bin/env python
# coding: utf-8

# # Objective : Big Mart Sales Prediction
# The aim is to build a predictive model and predict the sales of each product at a particular outlet.

# # Libraries

# In[1]:


# Python Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import seaborn as sns
import matplotlib as plt

# Metrices
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[49]:


# Load Data
train_data= pd.read_csv('train_v9rqX0R.csv')
test_data = pd.read_csv('test_AbJTz2l.csv')

print(train_data.shape, test_data.shape)


# In[3]:


train_data.head()


# In[4]:


train_data.duplicated().sum()


# In[21]:


train_data.info()


# In[6]:


train_data.isna().sum()


# Observation
# 1. rows: 8523 Columns:12
# 2. Target Column - Item_Outlet_Sales
# 3. Missing Data -  Item_Weight(1463) and Outlet_Size(2410)
# 4. Data Type: float64 - 4, int64- 1 and Object - 7

# In[7]:


train_data.describe().T


# In[8]:


# Data Distribution
train_data.hist(figsize=(10,10))


# In[9]:


# Correlation
train_data.corr()


# In[10]:


# Heat Map
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(train_data.corr(), annot=True)


# In[23]:


train_data.Item_Fat_Content.value_counts()


# In[71]:


# Replace LF, low fat with Low Fat and reg to Regular
train_data.Item_Fat_Content.replace(['LF','low fat'], value= 'Low Fat', inplace= True)
train_data.Item_Fat_Content.replace(['reg'], value= 'Regular', inplace= True)


# In[72]:


train_data["Item_Weight"].mean(), train_data["Item_Weight"].median()


# 1. train_data["Item_Weight"] - Filling null value with mean will be prone to outlier as its not normal distribution

# In[73]:


train_data["Item_Weight"].replace(to_replace=np.nan,value=round(train_data['Item_Weight'].median()),inplace=True)


# In[109]:


# Filling Null with mode : Feature- "Outlet_Size"
train_data["Outlet_Size"].fillna(value= "Medium",inplace=True)


# In[110]:


train_data["Outlet_Size"].value_counts()


# In[119]:


from datetime import date
today_date= date.today()
train_data['Years_Establishment'] = train_data.Outlet_Establishment_Year.apply(lambda x: today_date.year - x)
test_data['Years_Establishment'] = test_data.Outlet_Establishment_Year.apply(lambda x: today_date.year - x)


# In[120]:


train_data.drop(['Outlet_Establishment_Year'], axis = 1, inplace = True)
test_data.drop(['Outlet_Establishment_Year'], axis = 1, inplace = True)


# In[111]:


# Split the Data
from sklearn.model_selection import train_test_split

X= train_data.drop(['Item_Identifier','Item_Outlet_Sales'],axis=1)
y= train_data['Item_Outlet_Sales']


# In[112]:


X.Outlet_Size.isnull().sum()


# In[113]:


X.select_dtypes(include=object).columns


# In[114]:


# Label Encoder - Training Data & Test Data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# one hot encode input variables
le= LabelEncoder()

for feature in X.columns: # Loop through all columns in the dataframe
    if X[feature].dtype == object: # Only apply for columns with categorical strings
        X[feature]= le.fit_transform(X[feature])


# In[20]:


# Box Plot for Outliers
for columns in train_data:
    if train_data[columns].dtypes == object:
        print(columns)
        
    else:    
        plt.figure()
        train_data.boxplot([columns])


# In[117]:


# Split train and test Data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 7)# Stratify y to keep the class proportions consistent
# Print the number of train and test records.
print("Train Data:",X_train.shape)
print("Test Data:",X_test.shape)


# # Model Building

# In[151]:


# Linear Regression
from sklearn.linear_model import LinearRegression

lr= LinearRegression()
lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

lr_r2score = r2_score(y_test, lr_prediction)
lr_mae = mean_absolute_error(y_test, lr_prediction)
lr_mse = mean_squared_error(y_test, lr_prediction)

print(lr_r2score, lr_mae , lr_mse)


# In[126]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)

rf_r2score = r2_score(y_test, predictions)
rf_mae = mean_absolute_error(y_test, predictions)
rf_mse = mean_squared_error(y_test, predictions)

print(rf_r2score, rf_mae , rf_mse)


# In[122]:


# XGBoost
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000)
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(X_train, y_train, verbose=False)
predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, predictions)
xgb_mse = mean_squared_error(y_test, predictions)
xgb_r2score = r2_score(y_test, predictions)

print(xgb_r2score, xgb_mae , xgb_mse)


# # Hyper-parameter tuning

# In[123]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

criterion = ['gini', 'entropy']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[124]:


# Fetch the best parameters
from sklearn.ensemble import RandomForestRegressor
rf_model2 = RandomForestRegressor(random_state=1)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores

from sklearn.model_selection import RandomizedSearchCV
rf_random2 = RandomizedSearchCV(estimator = rf_model2, param_distributions = random_grid, n_iter = 10, 
                               cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random2.fit(X_train, y_train)
# fetch the best parameters
params = rf_random2.best_params_
params


# In[125]:


rf_model2 = RandomForestRegressor(n_estimators= params['n_estimators'],min_samples_split= params['min_samples_split'],
                                  min_samples_leaf=params['min_samples_leaf'],max_features= params['max_features'], 
                                  max_depth= params['max_depth'], bootstrap= True)

rf_model2.fit(X_train, y_train)
predictions = rf_model2.predict(X_test)


rf2_r2score = r2_score(y_test, predictions)
rf2_mae = mean_absolute_error(y_test, predictions)
rf2_mse = mean_squared_error(y_test, predictions)
print(rf2_r2score,rf2_mae,rf2_mse)


# In[152]:


# xgb_r2score, xgb_mae , xgb_mse - xgb_model
# rf_r2score, rf_mae , rf_mse - rf_model
# rf2_r2score,rf2_mae,rf2_mse - rf_model2
# lr_r2score, lr_mae , lr_mse
MAE= [lr_mae, rf_mae, rf2_mae, xgb_mae]
MSE= [lr_mse, rf_mse, rf2_mse, xgb_mse]
R_score= [lr_r2score, rf_r2score, rf2_r2score, xgb_r2score]
# Cross_score= [LR_CS,RFR_CS,LS_CS,XGB_CS,RD_CS]
Models = pd.DataFrame({
 'Models': ["Linear Regression","Random Forest Regression","Random Forest Regressor_Parameter tuning","XGB Regressor"],
 'MAE': MAE, 'MSE': MSE, 'R^2':R_score})
Models.sort_values(by='MAE', ascending=True)


# # Remove outlier/ setting the upper and lower limit

# In[131]:


# Quartile
Q1= train_data.Item_Visibility.quantile(0.25) #anything above is outlier
Q3 = train_data.Item_Visibility.quantile(0.75)
print(Q1, Q3)


# In[132]:


IQR = Q3- Q1


# In[133]:


lower_limit = Q1- 1.5* IQR
upper_limit = Q3+  1.5* IQR


# In[135]:


train_data[(train_data.Item_Visibility < lower_limit)| (train_data.Item_Visibility > upper_limit)]


# In[136]:


# Remove Outliers
train_data_nooutlier= train_data[(train_data.Item_Visibility > lower_limit)& (train_data.Item_Visibility < upper_limit)]


# In[137]:


train_data_nooutlier.head()


# In[142]:


train_data_nooutlier["Item_Weight"].value_counts()


# In[145]:


# Split the Data
from sklearn.model_selection import train_test_split

X_outlier= train_data_nooutlier.drop(['Item_Identifier','Item_Outlet_Sales'],axis=1)
y_outlier= train_data_nooutlier['Item_Outlet_Sales']


# In[146]:


# Split train and test Data
X_train_O, X_test_O, y_train_O, y_test_O = train_test_split(X_outlier, y_outlier,random_state = 7)# Stratify y to keep the class proportions consistent
# Print the number of train and test records.
print("Train Data:",X_train_O.shape)
print("Test Data:",X_test_O.shape)


# In[148]:


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train_O, y_train_O)
rf_prediction_O = rf_model.predict(X_test_O)
rf_r2score_O = r2_score(y_test_O, rf_prediction_O)
print(rf_r2score_O)


# In[150]:


from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000)
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(X_train_O, y_train_O, verbose=False)
xgb_prediction_O = xgb_model.predict(X_test_O)

mae = mean_absolute_error(xgb_prediction_O, y_test_O)
xgb_r2score_O = r2_score(y_test_O, xgb_prediction_O)
print(xgb_r2score_O)


# 1. Even after removing Outliers no improvement in the performance seen
# without Outlier - 
# with outlier- 

# In[ ]:




