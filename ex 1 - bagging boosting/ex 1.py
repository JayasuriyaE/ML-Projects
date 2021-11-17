import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score

#----------------------------------------------------------reading the dataset 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(url, names = names)
print(dataframe.head())
data = dataframe.values


#---------------------------------------------------------Splitting dattset 
X,y = data[:,:-1],data[:,-1]
print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 1)
print( X_train.shape , X_test.shape , y_train.shape , y_test.shape)


#----------------------------------------------------------Building the models
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(f"Decision tree Classifier accuracy on test data: {accuracy_score(y_test,y_pred)}")

##___________BAGGING_______________
bg = BaggingClassifier(dt,max_samples = 0.5,max_features = 1.0, n_estimators=20)
bg.fit(X_train,y_train)
y_pred = bg.predict(X_test)
print(f"Bagging Classifier testing accuracy: {accuracy_score(y_test,y_pred)}")

##___________BOOSTING_______________
adb = AdaBoostClassifier(dt,n_estimators = 10 , learning_rate = 1)
adb.fit(X_train,y_train)
y_pred = adb.predict(X_test)
print(f"AdaBoost Classifier testing accuracy: {accuracy_score(y_test,y_pred)}")


bg_cv = cross_val_score(bg,X_train,y_train,cv = 4)
print(f"\nBagging Classifier Cross Validation Score:{bg_cv.mean()}")

ada_cv = cross_val_score(adb,X_train,y_train,cv = 4)
print(f"AdaBoost Classifier Cross Validation Score:{ada_cv.mean()}")
