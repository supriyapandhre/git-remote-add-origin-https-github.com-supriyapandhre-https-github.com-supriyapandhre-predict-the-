#!/usr/bin/env python
# coding: utf-8

# In[81]:



import numpy as np
import pandas as pd 
import datetime
import os
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
print(os.getcwd())
os . chdir('D:/New folder2')


# In[82]:


import pandas as pd
file_name = 'D:/New folder2/Maize.xlsx' 


# In[83]:


file_name


# In[84]:


df = pd.read_excel(file_name, index_col=0)
print(df.head()) # print the first 5 rows


# In[85]:


df


# In[86]:


# Statistics for each column
df.describe()


# In[87]:


df.select_dtypes(include=['object']).head()


# In[88]:


# # Data Types and Missing Values

# See the column data types and non-missing values
df.info()


# In[89]:


df.shape


# In[90]:


df.info


# In[91]:


plt.subplots(figsize=(12,9))
sns.distplot(df['Price In India'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(df['Price In India'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel(' Years')

#Probablity plot

fig = plt.figure()
stats.probplot(df['Price In India'], plot=plt)
plt.show()


# In[92]:


#descriptive statistics summary
df['Price In India'].describe()


# In[93]:


#histogram
sns.distplot(df['Price In India']);


# In[94]:


#skewness and kurtosis
print("Skewness: %f" % df['Price In India'].skew())
print("Kurtosis: %f" % df['Price In India'].kurt())


# In[95]:


corr_matrix = df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr_matrix, vmax=0.9, square=True)


# In[96]:


sns.set()
cols = ['Price In India', 'Years','Quaters']
sns.pairplot(df[cols], size = 2.5)
plt.show();


# In[97]:


#applying log transformation
df['Price In India'] = np.log(df['Price In India'])


# In[98]:


#transformed histogram and normal probability plot
sns.distplot(df['Price In India'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['Price In India'], plot=plt)


# In[40]:


#Take targate variable into y
y = df['Price In India']


# In[41]:


#Delete the saleprice
del df['Price In India']


# In[42]:


#Take their values in X and y
X = df.values
y = y.values


# In[43]:


# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[ ]:



#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()

#Fit the model
model.fit(X_train, y_train)

#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))

#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[46]:


#Train the model
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)

#Fit
GBR.fit(X_train, y_train)

print("Accuracy --> ", GBR.score(X_test, y_test)*100)


# In[47]:





# In[ ]:




