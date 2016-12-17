#polynomial Regression
#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

#fit SVR regression
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#predict
yprdct=sc_y.inverse_transform(regressor.predict(np.array(sc_x.fit_transform([[6.5]]))))
print(yprdct)
#plot: SVR
xgrid=np.arange(min(x),max(x),0.1)
xgrid=xgrid.reshape(len(xgrid),1)
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(xgrid), sc_y.inverse_transform(regressor.predict(xgrid)), color='blue')
plt.title('SVR')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()