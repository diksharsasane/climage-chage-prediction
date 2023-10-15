# -*- coding: utf-8 -*-
"""Train

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zauee6ki2TE-DKVt5DgaDLot5NTZJaxc
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# %matplotlib inline

global_country = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\climate\\GlobalLandTemperaturesByCountry.csv')

#global_country.head()

#global_country.shape

sort_by_temp_desc = global_country.sort_values('AverageTemperature', ascending=False)
#sort_by_temp_desc

global_temp = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\climate\\GlobalTemperatures.csv')

global_temp = global_temp[['dt', 'LandAndOceanAverageTemperature']]
global_temp.dropna(inplace=True)
global_temp['dt'] = pd.to_datetime(global_temp.dt).dt.strftime('%d/%m/%Y')
global_temp['dt'] = global_temp['dt'].apply(lambda x:x[6:])
global_temp = global_temp.groupby(['dt'])['LandAndOceanAverageTemperature'].mean().reset_index()

plt.figure(figsize =(16, 6))
ax = sns.lineplot(
    x = global_temp['dt'],
    y = global_temp['LandAndOceanAverageTemperature'])
ax.set_title('Average Global Temperature Movement')
ax.set_ylabel('Average Global Temperature')
ax.set_xlabel('Date (1750 - 2015)')
ax.axes.get_xaxis().set_ticks([])
#ax

#global_temp

X = global_temp.iloc[:, global_temp.columns != 'LandAndOceanAverageTemperature'].values
y = global_temp.iloc[:, global_temp.columns == 'LandAndOceanAverageTemperature'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#X_train

# Fitting a linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

X_train = X_train.astype('float64')
y_train = y_train.astype('float64')
X_test = X_test.astype('float64')
y_test = y_test.astype('float64')

# X_train = X_train.flatten()
# y_train = y_train.flatten()

# # Convert X_test and y_test to 1D arrays or lists
# X_test = X_test.flatten()
# y_test = y_test.flatten()




plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Average Global Temperature (Training Set)')
plt.xlabel('Year')
plt.ylabel('Temperatre (c)')
plt.show()

# Visualising the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Average Global Temperature (Test set)')
plt.xlabel('Year')
plt.ylabel('Temperature(c)')
plt.show()


def predict_temperature(year):
     
    # Code for making temperature predictions
     X_predict =year # Enter the year that you require the temperature for
     X_predict = np.array(X_predict).reshape(1, -1)#
     y_predict = regressor.predict(X_predict)
     return y_predict



# # Building the predictor
# X_predict = [2050] # Enter the year that you require the temperature for
# X_predict = np.array(X_predict).reshape(1, -1)#
# y_predict = regressor.predict(X_predict)

# # Outputting the predicted temperature of the year above
# y_predict