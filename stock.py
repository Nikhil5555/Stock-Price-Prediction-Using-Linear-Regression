import quandl
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
#replacement for cross validation
from sklearn.model_selection import train_test_split

quandl.ApiConfig.api_key = "Jj9Bi5HKgzVkeubtq7dU"
df = quandl.get("WIKI/AMZN")

df = df[['Adj. Close']]

forecast_out = int(30)

df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))


X_forecast = X[-forecast_out:]
X = X[:-forecast_out]
#print(X_forecast)
y = np.array(df['Prediction'])
y = y[:-forecast_out]
# below sentencce has been modified for train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
# print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)
#print(forecast_prediction)
#Now lets predict for another 30 days
X2=pd.DataFrame(X)
X_forecast2=pd.DataFrame(X_forecast)
result=X2.append(X_forecast2)
y2=pd.DataFrame(y)
y_forecast=pd.DataFrame(forecast_prediction)
result2=y2.append(y_forecast)
X_train, X_test, y_train, y_test = train_test_split(result, result2, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)
k=[]
l=1588
for i in range(30):
    forecast_prediction = clf.predict([[l]]).reshape(-1,1)
    l=forecast_prediction[0][0]
    print(forecast_prediction[0][0])
    #We should not use this model for predicting price for more days as it 
    #could be wrong beacuse of simple features 
    #you can use more features to get a good model than this