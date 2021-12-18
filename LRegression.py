#!/user/nickmathein/miniforge/lib/python3.9 python

#Rasmussen Mathein Data extraction
import yfinance as yh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


from yfinance import ticker

def Collect_Data(ticker):
    stock = yh.Ticker(ticker)
    history = stock.history(period='max')
    history = history[["Close","Volume"]]
    ema10 = history.ewm(span=10).mean()
    ema10 = ema10.rename(columns={"Close":"EMA"})
    ema = ema10["EMA"]
    data = history.join(ema)
    return data




tticker = 'AAPL'
df = Collect_Data(tticker)

X_train, X_test, y_train, y_test = train_test_split(df[['Close']], df[['EMA']], test_size=.2)



model = LinearRegression()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

predicted_price = pd.DataFrame(y_predict,index=y_test.index,columns = ['Close'])  
# predicted_price['Close'].plot(color = 'blue', title='Prediction Price')  
predicted_price['Close'].plot(color='orange')
df['Close'].plot(label=ticker, figsize=(15, 9), title='Linear Regression Model for OPTT'.format(ticker), color='red', linewidth=1.0, grid=True)
df['EMA'].plot(label=ticker, figsize=(15, 9), title='Linear Regression Model for OPTT'.format(ticker), color='black', linewidth=1.0, grid=True)

plt.legend(['Predicted Close', 'Actual Close', 'EMA'])
plt.show()
print(mean_squared_error(y_test, y_predict))



y_predict = model.predict(X_test)

predicted_price = pd.DataFrame(y_predict,index=y_test.index,columns = ['Close'])  
# predicted_price['Close'].plot(color = 'blue', title='Prediction Price')  
predicted_price['Close'].plot(color='orange')
df['Close'].plot(label=ticker, figsize=(15, 9), title='Linear Regression Model for OPTT'.format(ticker), color='red', linewidth=1.0, grid=True)
df['EMA'].plot(label=ticker, figsize=(15, 9), title='Linear Regression Model for OPTT'.format(ticker), color='black', linewidth=1.0, grid=True)




