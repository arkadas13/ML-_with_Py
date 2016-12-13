#simple linear regression

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#splitting dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=1/3, random_state=0)

#Simple linear regression
#fitting
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
#predict
y_prdct=regressor.predict(x_test)
#visualise training set
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train,y_train, color='red')
ax1.plot(x_train, regressor.predict(x_train), color='blue')
ax1.set_title('Salary vs Experience (training set)')
ax1.set_xlabel('Years of Experience')
ax1.set_ylabel('Salary')

#visualise test set
ax2.scatter(x_test,y_test, color='red')
ax2.plot(x_train, regressor.predict(x_train), color='blue')
ax2.set_title('Salary vs Experience (test set)')
ax2.set_xlabel('Years of Experience')
ax2.set_ylabel('Salary')
plt.show()