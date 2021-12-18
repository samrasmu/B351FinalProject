# B351FinalProject
#Sam Rasmussen and Nick Mathein B351 Final Project

#Description of Project and Installation:

#Python3 must be installed on your device in order to properly run the code in this repository.

##Linear Regression Script##
"""
#The beginning of the script imports all necesarry libraries to create a linear regression model. These libraries include functions from sklearn that allow for metrics to be calculated like the exponential moving average(EMA) and mean standard error. 

On line 16 the script defines a key function that accesses the yfinance library and pulls in the historical data for a given ticker. This data is put into a pandas dataframe and returned to the program.

Next the ticker is assigned and the training data and testing data are split up to create and test the model. 

After the model is defined using sklearns linear regression model. Which allows training data to be used from a pandas DataFrame.

The training data is used to create the model.

Then we iterate through the stock tickers and create graphs that have the actual price the predicted price and the EMA for a given day.

Print the Mean standard error for each as well.
"""

#LSTM RNN Model
"""
#The code includes all the neccesary libraries you will need to perform the Long-Short Term Memory Recurrent Neural Network Model. The libraries include: SKLearn, Keras, YFinance, Pandas, Tensorflow.Keras, and MatPlotLib. These are all necessary to perform the functions of the model, the collection and reading of data, and the visualization of the graphs.

Overall, the Long-Short Term Memory RNN provides a representation of a Next-Day Closing Price Stock Predictor.

The coding process began with the declaration of the time periods for the training and testing data followed by a function, Collect_Data, which retrieved historical data from the library YFinance. We used the MinMaxScaler function in the SKLearn library in order to make the data more normal and scale it in-between 0 and 1. Since we are working with a LSTM RNN, we had to add to our array so we created a loop to make the data more concise for it to go through our model. Then we went through the actual model with there layers. For this part you must have TensorFlow and Keras installed. Finally the actual predictions are made by calling the prediction on the testing data we included. 


"""
