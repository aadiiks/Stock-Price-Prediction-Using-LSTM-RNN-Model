#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # To avoid Future Wrnings


# In[]:


start = '2012-01-01'
end = '2021-12-31'
#data = data.DataReader(name="TSLA", data_source='yahoo', start=start_date, end=end_date)
data = web.DataReader('CL=F', 'yahoo', start, end)


# In[]:


data.info()


# In[]:


data.isnull().sum()


# In[]:


data.to_csv('CL=F10.csv')


# In[]:


start = '2022-01-01'
end = '2022-04-30'
#data = data.DataReader(name="TSLA", data_source='yahoo', start=start_date, end=end_date)
train_set = web.DataReader('IOC.NS', 'yahoo', start, end)


# In[]:


train_set.info()


# In[]:


train_set['Volume'] = train_set['Volume'].astype(float)


# In[]:


train_set.info()


# In[]:


train_set.isnull().sum()


# In[]:


train_set.to_csv('IOC_JAN_FEB_MAR_APR.NS.csv')
