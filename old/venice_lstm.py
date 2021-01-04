
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np# linear algebra
from numpy import nan
from numpy import array
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure


# In[2]:


#already dropped NA's
#sorted entries by date
dataset = pd.read_csv('venice_new.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

print(dataset.tail())


# # Preprocessing data 
# https://machinelearningmastery.com/how-to-load-and-explore-household-electricity-usage-data/

# In[3]:


# line plot for each variable  for the whole time range
plt.figure()
for i in range(len(dataset.columns)):
	plt.subplot(len(dataset.columns), 1, i+1)
	name = dataset.columns[i]
	plt.plot(dataset[name])
	plt.title(name, y=0)
plt.show()


# In[6]:


# line plot for each variable  foronly one year
dataset_19= dataset.loc[dataset.index.year==2019]

plt.figure()

for i in range(len(dataset_19.columns)):
	plt.subplot(len(dataset_19.columns), 1, i+1)
	name = dataset_19.columns[i]
	plt.plot(dataset_19[name])
	plt.title(name, y=0)
plt.show()


# In[7]:


# line plot for each variable  foronly one year
dataset_y= dataset.loc[dataset.index.year==2020]

plt.figure()

for i in range(len(dataset_y.columns)):
	plt.subplot(len(dataset_y.columns), 1, i+1)
	name = dataset_y.columns[i]
	plt.plot(dataset_y[name])
	plt.title(name, y=0)
plt.show()


# In[8]:


# plot active power for each year
months = [x for x in range(9, 11)]
plt.figure()
for i in range(len(months)):
	# prepare subplot
	ax = plt.subplot(len(months), 1, i+1)
	# determine the month to plot
	month = '2019-' + str(months[i])
	# get all observations for the month
	result = dataset_19[month]
	# plot the active power for the month
	plt.plot(result['temp'])
	# add a title to the subplot
	plt.title(month, y=0, loc='left')
plt.show()


# In[9]:


#print(dataset_19[month].to_string())


# In[10]:


# histogram plot for each variable
#wh is stronly gaussian 
#temperature more periodical-probably due to the seasons of the year
plt.figure()
for i in range(len(dataset.columns)):
	plt.subplot(len(dataset.columns), 1, i+1)
	name = dataset.columns[i]
	dataset[name].hist(bins=100)
	plt.title(name, y=0)
plt.show()


# # Problem framing
# I guess we want to for example predict the temperature and water level for the next month
# 

# # Approach
# Classical linear methods include techniques are very effective for univariate time series forecasting. SARIMA
# or
# Deep Learning CNN LSTM and ConvLSTM

# # Try DL LSTM of course 
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
# #maybe group the data by day
# #then split the data into week chunks
# 

# In[11]:


len(dataset_19)#11940
len(dataset_y)#1400 
#It seems that they measure more when there are more extreme values


# In[3]:


daily_groups = dataset.resample('D')
daily_data = daily_groups.mean()
#print(daily_data.to_string())


# In[13]:


plt.figure()
for i in range(len(daily_data.columns)):
	plt.subplot(len(daily_data.columns), 1, i+1)
	name = daily_data.columns[i]
	plt.plot(daily_data[name])
	plt.title(name, y=0)
plt.show()


# In[14]:


#Even though there are only a few periods where there are daily measurements. I will interpolate the data. 


# In[4]:


# Interpolate the dataset based on previous/next values..
#This works super good. it's like a gradual inference of the data points
def impute_interpolate(dataset, col):
    dataset[col] = dataset[col].interpolate()
    # And fill the initial data points if needed:
    dataset[col] = dataset[col].fillna(method='bfill')
    return dataset


# In[5]:


data= impute_interpolate(dataset=dataset, col='wh')
data= impute_interpolate(dataset=data, col='temp')


# In[7]:


daily_data= data
#print(daily_data.to_string())


# In[18]:


plt.figure()
for i in range(len(daily_data.columns)):
	plt.subplot(len(daily_data.columns), 1, i+1)
	name = daily_data.columns[i]
	plt.plot(daily_data[name])
	plt.title(name, y=0)
plt.show()


# # Outlier Detection and Imputation 

# In[19]:


import numpy as np
from pykalman import KalmanFilter

# Implements the Kalman filter for single columns.
class KalmanFilters:

    # Very simple Kalman filter: fill missing values and remove outliers for single attribute.
    # We assume a very simple transition matrix, namely simply a [[1]]. It
    # is however still useful as it is able to dampen outliers and impute missing values. The new
    # values are appended in a new column.
    def apply_kalman_filter(self, data_table, col):

        # Initialize the Kalman filter with the trivial transition and observation matrices.
        kf = KalmanFilter(transition_matrices = [[1]], observation_matrices = [[1]])

        numpy_array_state = data_table.as_matrix(columns=[col])
        numpy_array_state = numpy_array_state.astype(np.float32)
        numpy_matrix_state_with_mask = np.ma.masked_invalid(numpy_array_state)

        # Find the best other parameters based on the data (e.g. Q)
        kf = kf.em(numpy_matrix_state_with_mask, n_iter=5)

        # And apply the filter.
        (new_data, filtered_state_covariances) = kf.filter(numpy_matrix_state_with_mask)

        data_table[col + '_kalman'] = new_data
        return data_table


# In[20]:


# # Kalman Filter 
#     # Very simple Kalman filter: fill missing values and remove outliers for single attribute.
#     # We assume a very simple transition matrix, namely simply a [[1]]. It
#     # is however still useful as it is able to dampen outliers and impute missing values. The new
#     # values are appended in a new column.
    
# #K= KalmanFilters()
# data= K.apply_kalman_filter(data_table=data, col='wh')
# data= K.apply_kalman_filter(data_table=data, col='temp')
# #print(data)


# In[21]:


plt.figure()
for i in range(len(data.columns)):
	plt.subplot(len(data.columns), 1, i+1)
	name = data.columns[i]
	plt.plot(data[name])
	plt.title(name, y=0)
plt.show()


# In[22]:


# ok so basically using the kalman diter didnt have much of an effect, at least based on v rough visual inspection.
#Mostly smoothes values. Could have also used kalman instead of interpolation. 


# #Prepare data for Learning

# In[23]:


# # now we just split it off. but actually we should take random weeks 
# dataset=data
# training_frac=0.7
# features=':'
# end_training_set = int(training_frac * len(dataset.index))
# training_set_X = dataset.ix[0:end_training_set]
# training_set_y = dataset.ix[0:end_training_set]
# test_set_X = dataset.ix[end_training_set:len(dataset.index)]
# test_set_y = dataset.ix[end_training_set:len(dataset.index)]
# print(len(test_set_X))
# print(len(training_set_X))



# In[7]:


# split a univariate dataset into train/test sets
def split_dataset(data):
	data= data[:(int(len(data)/7))]
	test_size= int(len(data)*0.3)
	train, test = data[1:-test_size], data[-test_size:-6]
	# restructure into windows of weekly data
	train= np.array(train) #if we dont convert here, it doesnt work or a list of dataframes comes out (without the second np.array)
	test= np.array(test)
	train = np.array(np.split(train, len(train)/7)) # if we dont put the np.array here a list of numpy arrays comes out
	test = np.array(np.split(test, len(test)/7))
	return train, test


# In[ ]:


train, test = split_dataset(data.values)
# validate train data
print(train.shape)
# validate test
print(test.shape)


# In[9]:


#so we put in a week and try to get out a week
data_X, data_y = to_supervised(train, n_input=7, n_out=7)


# In[29]:


data_X.shape


# In[10]:


#this makes the sliding windows I think 
# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)


# In[15]:


# univariate multi-step lstm
from math import sqrt
 
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM


# In[11]:


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
 
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))


# In[12]:


# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 70, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
 
# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
 
# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores


# In[16]:


# # load the new file
# dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# # split into train and test
# train, test = split_dataset(dataset.values)
# # evaluate model and get scores
n_input = 7
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()


# In[17]:


print(score, scores)


# ## Now I will try to do it in Pytorch
# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

# In[44]:


import torch
import torch.nn as nn

import seaborn as sns



# In[63]:


test_data_size = round(len(dataset)*0.3) 
print(test_data_size)

train_data = dataset[:-test_data_size]
test_data = dataset[-test_data_size:]

train_data= np.array(train_data)


# In[64]:


train_data_normalized = torch.FloatTensor(train_data).view(-1) #I think the flattenign was not necessart
train_data_normalized.shape


# In[66]:


#print(train_data_normalized[:5])
print(train_inout_seq[-5:])


# In[50]:


#returns a list of tuples
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]  # the label is sort o f the predicted. so all of of the predictions become one time longet now? 
        inout_seq.append((train_seq ,train_label))  #tuple containing the 7 days values and the next value 
    return inout_seq


# In[65]:


train_window=7
train_inout_seq = create_inout_sequences(train_data, train_window) #


# In[58]:




