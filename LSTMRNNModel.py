import yfinance as yh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#mean-square-error bar graph imports
from sklearn.metrics import mean_squared_error


#references:
# https://www.nbshare.io/notebook/249468051/How-To-Code-RNN-and-LSTM-Neural-Networks-in-Python/
# https://www.thecodehaven.com/ai-predicting-stock-prices-2021-easy-steps-with-code/
#

#training data start and end time 
start = "2017-05-01"
end = "2021-01-01"

#test data start and end times
test_start = "2021-01-02"
test_end = "2021-08-02"

def Collect_Data(ticker):
    data = yh.download(ticker, start, end, interval='1d')
    return data 
data = Collect_Data("TSLA")  

data['Close'].plot()
plt.show()
past_days = 150 #how many days in the past you want to see
training = data['Close'].values.reshape(-1, 1)

sc = MinMaxScaler(feature_range=(0,1))
scaled_training = sc.fit_transform(training)

x_train = []
y_train = []
num = len(scaled_training)

for i in range(past_days, num):
    x_train.append(scaled_training[i-past_days:i, 0])
    y_train.append(scaled_training[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#must reshape data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#rnn model for actually training -> sequential serves as a linear stack layer
regressor = Sequential()
#3 layers of the LSTM rnn 50 neurons in each which represents the dimensions 
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#LSTM layer with 20% dropout when it moves to following layer;
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#Dense layer, which makes the data into one value
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train, y_train, epochs = 25, batch_size=32)
#actual price
test_data = Collect_Data("TSLA")
real_price = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
inputoregressor= total_dataset[len(total_dataset)-len(test_data)-past_days:].values.reshape(-1, 1)
inputoregressor = sc.transform(inputoregressor)

x_test = []
for j in range(past_days, len(inputoregressor)):
    x_test.append(inputoregressor[j-past_days:j, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictedprice = regressor.predict(x_test)
predictedprice = sc.inverse_transform(predictedprice)

#visualization -> graphs w/ actual stock price line and predicted stock price line
plt.plot(real_price, color="black", label=f"Actual GE Price")
plt.plot(predictedprice, color='green', label=f"Predicted GME Price")
plt.title(f"GE" )
plt.xlabel('Time')
plt.ylabel(f'GE Share Price')
plt.legend()
plt.show()

#displaying predicted price value
real_data = [inputoregressor[len(inputoregressor)  - past_days:len(inputoregressor+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))
prediction = regressor.predict(real_data)
prediction = sc.inverse_transform(prediction)
print(f"Prediction: {prediction}")

#mean-square-error implementation
print(mean_squared_error(real_price, predictedprice))
#actual next day price value
#TSLA: 
# Prediction: $709.74
#training data start and end time 
# start = "2017-05-01"
# end = "2021-01-01"

#test data start and end times
# test_start = "2021-01-02"
# test_end = "2021-08-02"

#GME: 
#Prediction: [[193.39351]]
#1812.4651226539129
# #training data start and end time 
# start = "2019-01-01"
# end = "2021-08-01"

# #test data start and end times
# test_start = "2021-09-01"
# test_end = "2021-12-02"

#GE:
# Prediction: [[78.79599]]
# 29.871564402513087
# #training data start and end time 
# start = "2019-01-01"
# end = "2021-01-01"

# #test data start and end times
# test_start = "2021-01-02"
# test_end = "2021-12-16"
######