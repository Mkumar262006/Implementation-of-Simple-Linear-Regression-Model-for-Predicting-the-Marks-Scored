# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MANOJ KUMAR S
RegisterNumber:  212223240082
*/
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()
df.tail()
#segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
Y_pred
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print('RMSE= ',rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![Screenshot 2024-02-19 144550](https://github.com/Mkumar262006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139472/50f6d2a3-49c6-46ce-905f-2b7f9a68a5ec)
![Screenshot 2024-02-19 144620](https://github.com/Mkumar262006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139472/6b01571b-8477-4d63-bf7a-b75b67df91c3)
![Screenshot 2024-02-19 144636](https://github.com/Mkumar262006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139472/0fdab683-02f4-44d7-be91-a828a638ab4c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
