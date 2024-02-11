#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# In[3]:


#read the data
df1=pd.read_csv('AirPassengers.csv')


# In[4]:


#check datatypes
df1.dtypes


# In[5]:


#we are providing inputs to tell pandas that we are trying to work with time series.
df1=pd.read_csv('AirPassengers.csv',parse_dates=['Month'])


# In[6]:


df1.dtypes


# In[7]:


df1.head()


# In[8]:


#it is recommended that we make our time series refrence as the index
df1=pd.read_csv('AirPassengers.csv',parse_dates=['Month'],index_col='Month')


# In[9]:


df1.head()


# In[10]:


#slicing of data
df1['1951-04-01':'1952-03-01']


# In[11]:


#we can check values corresponding to a specific time point
df1.loc['1960-05-01']
#loc is uesd for exact value


# In[12]:


#plot the time series
df1.plot()
plt.show()
#here there is trend in graph but there is no sesonality


# In[14]:


#increase the figure size
from pylab import rcParams
rcParams['figure.figsize']=12,8
df1.plot()
plt.show()


# In[16]:


#decompose the time series multiplicatevely
df1_mul_decompose=seasonal_decompose(df1,model="multiplicative")
df1_mul_decompose.plot()
plt.show()


# In[17]:


#try to do log transformation
df1_log=df1.copy()


# In[19]:


df1_log['Passengers']=np.log(df1)


# In[20]:


df1_log.Passengers


# In[21]:


#visualize
df1_log.plot()
plt.show()


# In[22]:


#Compare with the original series
plt.subplot(2,1,1)
plt.title('Original Time Series')
plt.plot(df1)

plt.subplot(2,1,2)
plt.title('Log transformed Time Series')
plt.plot(df1_log)
plt.tight_layout


# In[ ]:




