from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/GrandmaCan/ML/main/Classification/Diabetes_Data.csv'
data = pd.read_csv(url)

data['Gender'] = data['Gender'].map({"男生":1,"女生":0})


X = data[['Age','Weight','BloodSugar','Gender']]
y = data['Diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 87)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Logistic Regression
lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)
y_pred = lg_model.predict(X_test)
print((y_pred == y_test).sum()/ len(y_test))

# SVM Model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print((y_pred == y_test).sum()/ len(y_test))