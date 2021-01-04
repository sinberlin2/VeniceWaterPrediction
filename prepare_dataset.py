import pandas as pd
import numpy as np# linear algebra
from numpy import nan
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure


# df1 = pd.read_csv('VeneziaPuntaSalute.csv', delimiter=',')
# df1.columns = ['date','time','liv','temp']
#
# df1.time = df1.time.map(lambda x: x.rstrip('AMP').replace(';',''))
# df1.temp = df1.temp.map(lambda x: x.replace(';','')).replace(r'^\s*$', np.nan, regex=True).fillna(0).astype(float)
# df1.liv = df1.liv.fillna(0)
# df1.date = pd.to_datetime(df1.date)
#
# df1.time = pd.to_datetime(df1.time)
#
# #print(df1.head())


dataset = pd.read_csv('VeneziaPuntaSalute.csv', sep=',', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

dataset.columns = ['wh','temp']  #dont mix up tempory with temperature
#get rid of semicolons at the end
dataset.temp= dataset.temp.map(lambda x: x.replace(';','')).replace(r'^\s*$', np.nan, regex=True) .astype(float)

#inspect data
#print(dataset.head())

# find all missing values
#print(len(dataset.loc[dataset['wh'].isna()]))
print(dataset.loc[dataset['temp'].isna()])
#drop NAs
dataset = dataset.dropna()

#order the entries by the time
dataset=dataset.sort_index()


#dataset2= dataset.replace('NaN, 0, inplace=True)  # dataset= not necessary # this is for numpy array actually

# # save updated dataset
dataset.to_csv('venice_new.csv')
dataset=  pd.read_csv('venice_new.csv', index_col=0)


