#polynomial Regression
#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
#fit linear regression
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(x,y)
#fit polynomial regression
from sklearn.preprocessing import PolynomialFeatures
polyreg=PolynomialFeatures(degree=4)
xpoly=polyreg.fit_transform(x)
linreg2=LinearRegression()
linreg2.fit(xpoly,y)

#plot: Linear
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x,y, color='red')
ax1.plot(x, linreg.predict(x), color='blue')
ax1.set_title('Linear Regression')
ax1.set_xlabel('Position')
ax1.set_ylabel('Salary')

#plot: Polynomial
xgrid=np.arange(min(x),max(x),0.1)
xgrid=xgrid.reshape(len(xgrid),1)
ax2.scatter(x,y, color='red')
ax2.plot(xgrid, linreg2.predict(polyreg.fit_transform(xgrid)), color='blue')
ax2.set_title('Polynomial Regression')
ax2.set_xlabel('Position')
ax2.set_ylabel('Salary')
plt.show()

#predict: Linear
print(linreg.predict(x))

#predict: Polynomial
print(linreg2.predict(polyreg.fit_transform(x)))