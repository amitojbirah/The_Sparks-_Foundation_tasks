#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - Data Science and Business Analyatics Intership

# # Name: Amitoj Birah
# 
# ## Domain: DATA SCIENCE AND BUSINESS ANALYTICS
# 
# ## Task 1: PREDICTION USING SUPERVISED LEARNING
# 
# ## Language:Python
# 
# ## Dataset Link:http://bit.ly/w-data

# ## Importing libraries

# In[2]:


# Importing the required libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as ttsa

# To ignore the warning

import warnings as wg
wg.filterwarnings("ignore")


# ## Reading the Data 

# In[3]:


# Reading data from remote link

url='http://bit.ly/w-data'
data = pd.read_csv(url)
data


# In[4]:


data.head()


# In[5]:


data.tail()


# ## Analysis of the Data 

# In[8]:


# To find more information about our dataset
data.info()


# In[9]:


data.describe()


# # Visualizing the dataset

# In[10]:


# Plot comparing Hours studied with Test score 

data.plot(x='Hours',y='Scores', style='go')
plt.title('Prediction' )
plt.xlabel('Hours_Studied')
plt.ylabel('Test_score')
plt.show()


# In[11]:


## Box plot of the Dataset 


sns.boxplot(data=data[['Hours','Scores']])


# In[25]:


## Heat map of the Dataset 

sns.heatmap(data.corr(), annot= True,cmap='YlGnBu' )


# ## Preparing the Data

# In[12]:


# using iloc function we will divide the data
X = data.iloc[:,:-1].values
Y= data.iloc[:,1].values


# ## Splittting the Data into Training set and Test set

# In[15]:


## Splitting data into training and testing data
X_train,x_test,Y_train,y_test = tts(X,Y,test_size=0.20,random_state = 0)


# ## Training the algorithm

# In[16]:


## Lr() represents Linear regression
Reg = lr()
Reg.fit(X_train,Y_train)
print("Training complete.")


# ## Plotting the Predicted Data 

# In[17]:


# Plotting for the training data
L = Reg.coef_*X+ Reg.intercept_
data.plot.scatter(x='Hours', y='Scores')
plt.plot(X,L)
plt.grid()
plt.show()


# ## Comparing the Predictions 

# In[18]:


# Predicting the scores
y_pred=Reg.predict(x_test)
print(y_pred)
# Predicting the scores
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# ## final prediction for the case that if a student studies 9.25 hrs/day
# 

# In[19]:


h = np.array([[9.25]])
p = Reg.predict(h)
print('No of hours =', h[0][0])
print('Predicted Score =',p[0])


# ## Evaluating the model 

# In[21]:


from sklearn import metrics
print('Mean Absolute Error :',
     metrics.mean_absolute_error(y_test, y_pred))  # Evaluation by Mean_Absolute_Error Method
import math
print('Mean Square Error :',metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Square Error :',math.sqrt(metrics.mean_squared_error(y_test, y_pred)))

