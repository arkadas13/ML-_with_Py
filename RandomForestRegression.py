#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#fit Decision Tree regression
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x,y)

#predict
yprdct=regressor.predict(6.5)
print(yprdct)
#plot: SVR
xgrid=np.arange(min(x),max(x),0.01)
xgrid=xgrid.reshape(len(xgrid),1)
plt.scatter(x,y, color='red')
plt.plot(xgrid, regressor.predict(xgrid), color='blue')
plt.title('Random Forest')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()