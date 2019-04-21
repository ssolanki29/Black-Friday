#importing

import numpy as np
import pandas as pd
import matplotlib as plt

#preprocessing

Data_test = pd.read_csv('test.csv')
Data_train = pd.read_csv('train.csv')

X_train = Data_train.iloc[:,0:11]
Y_train = Data_train.iloc[:,-1]
X_test = Data_test.iloc[:,0:11]
pd.unique(Data_train).sum()
X_train.isnull().sum()
X_test.isnull().sum()

X_train.iloc[:,6] = X_train.iloc[:,6].replace({'4+' : 4})
X_test.iloc[:,6] = X_test.iloc[:,6].replace({'4+' : 4})

X_test = X_test.drop(['Product_Category_2','Product_Category_3'],axis=1)
X_train = X_train.drop(['Product_Category_2','Product_Category_3'],axis=1)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder = LabelEncoder()
labelencoder1 = LabelEncoder()


X_train.iloc[:,3] = labelencoder.fit_transform(X_train.iloc[:,3])
X_train.iloc[:,1] = labelencoder.fit_transform(X_train.iloc[:,1])
X_train.iloc[:,2] = labelencoder.fit_transform(X_train.iloc[:,2])
X_train.iloc[:,5] = labelencoder.fit_transform(X_train.iloc[:,5])
X_train.iloc[:,0] = labelencoder.fit_transform(X_train.iloc[:,0])

X_test.iloc[:,3] = labelencoder1.fit_transform(X_test.iloc[:,3])
X_test.iloc[:,1] = labelencoder1.fit_transform(X_test.iloc[:,1])
X_test.iloc[:,2] = labelencoder1.fit_transform(X_test.iloc[:,2])
X_test.iloc[:,5] = labelencoder1.fit_transform(X_test.iloc[:,5])
X_test.iloc[:,0] = labelencoder1.fit_transform(X_test.iloc[:,0])
X_test = pd.get_dummies(X_test,drop_first = True)
X_train = pd.get_dummies(X_train,drop_first = True)

#test.ix[test['Product_ID'].isin(new_product_ids), 'Product_ID'] = -1

#########

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)


#########cross validation
from sklearn.model_selection import cross_val_score
sc = cross_val_score(estimator= regressor,X = X_train,y = Y_train)
sc.mean()


from sklearn.model_selection import GridSearchCV

param = [{'n_estimators':[1,10,25,30,35,40,50,60,70,85,100]}]
grid = GridSearchCV(estimator = regressor,param_grid=param)
grid.fit(X_train,Y_train)

parameter = grid.best_params_
score = grid.best_score_


np.savetxt('SampleSubmission_TmnO39y',Y_pred,fmt='%0f')
 

