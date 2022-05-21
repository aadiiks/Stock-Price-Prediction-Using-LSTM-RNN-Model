#!/usr/bin/env python
# coding: utf-8

# # **Imported Libraries**

# In[]:


import seaborn as sns
sns.set()
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # To avoid Future Wrnings


# ## Reading the Dataset

# In[]:


# data = pd.read_csv('IOC10.NS.csv')


# In[]:


start = '2012-01-01'
end = '2021-12-31'
#data = data.DataReader(name="TSLA", data_source='yahoo', start=start_date, end=end_date)
data = web.DataReader('IOC.NS', 'yahoo', start, end)
data.head()


# In[]:


# data = pd.read_csv('D:\CODING\Stock_Prediction\IOC10.NS.csv')


# In[]:


data.head(10)


# In[]:


data.info()


# In[]:


data.isnull().sum() #Checking if there is any Null Values or not


# In[]:


data['Open'].plot(figsize=(16,6))


# # Replacing the Type to make the data Homogenenous

# In[]:


data['Close'] = data['Close'].str.replace(',','').astype(float)


# In[]:


data['Volume'] = data['Volume'].str.replace(',','').astype(float)


# In[]:


#7Day Rolling
data.rolling(7).mean().head(20)


# In[]:


data['Open'].plot(figsize=(16,6))
data.rolling(window=30).mean()['Close'].plot()


# In[]:


data['Close: 30 Day Mean']= data['Close'].rolling(window=30).mean()
data[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))


# In[]:


# Optional specify a minimum number of periods
data['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))


# # Data Preproccessing

# In[]:


training_set=data['Open']
training_set=pd.DataFrame(training_set)


# In[]:


# Feature Scalling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[]:


#Creating a Data Structure with a 60 Timesteps and 1 Output

X_train = []
y_train = []
for i in range(60, 2465):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[]:


# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[]:


# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(data.corr(), annot = True)

plt.show()


# # Feature Extraction

# In[]:


#Builiding the RNN Model
import keras
from keras.models import Sequential # Linear stack of layers to create sequential model
from keras.layers import Dense # Regular connected to neural network (change the dimensions of the vector obtained)
from keras.layers import LSTM
from keras.layers import Dropout


# In[]:


# Initialising the RNN
regressor = Sequential()


# In[]:


# Adding the first LSTM Layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


# In[]:


# Adding the second LSTM Layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# In[]:


# Adding the third LSTM Layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# In[]:


# Adding the fourth LSTM Layer and some Dropout regularisation

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[]:


# Adding the output Layer

regressor.add(Dense(units = 1))


# In[]:


# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# Fitting the RNN to the Training set
history=regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[]:


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()


# In[]:


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()


# In[:


# Making the prediction and visualising the results

# Getting the real stock price of JAN 2022

dataset_test = pd.read_csv('IOCjan.NS.csv',index_col="Date",parse_dates=True)
real_stock_price = dataset_test.iloc[:, 1:2].values


# In[]:


start = '2022-01-01'
end = '2022-01-31'
#data = data.DataReader(name="TSLA", data_source='yahoo', start=start_date, end=end_date)
dataset_test = web.DataReader('IOC.NS', 'yahoo', start, end)
real_stock_price = dataset_test.iloc[:, 1:2].values


# In[]:


dataset_test.head()


# In[]:


dataset_test.info()


# In[]:


# dataset_test['Volume'] = dataset_test['Volume'].str.replace(',','').astype(float)
dataset_test['Volume'] = dataset_test['Volume'].astype(float)


# In[]:


dataset_test.info()


# In[]:


test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)


# In[]:


#Getting the Predicted stock Prices of 2022

dataset = pd.concat((data['Open'], dataset_test['Open']), axis = 0)
inputs = dataset[len(dataset) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[]:


predicted_stock_price = pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()
#predicted_stock_price.head()


# In[]:


plt.plot(real_stock_price, color = 'red', label = 'Real IOC Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted IOC Stock Price')
plt.title('IOC Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IOC Stock Price')
plt.legend()
plt.show()


# In[]:


#Finished
predicted_stock_price.to_csv('Stock_Price_Prediction_IOC.csv')
