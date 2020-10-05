
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('bank.csv', header=0, delimiter=';')
df.head()
df.dropna()

print(list(df.columns))

df.drop(df.columns[[0, 3, 5, 8, 9, 10, 11, 12, 13, 14]], axis=1, inplace=True)
df.head()
data = pd.get_dummies(df, columns =['job', 'marital', 'default', 'housing',
'loan', 'poutcome'])
data.head()
data.columns
data.drop(data.columns[[12, 25]], axis=1, inplace=True)
data.columns
X = data.iloc[:,1:]
X.head()
Y = data.iloc[:,0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
classifier = LogisticRegression(solver="lbfgs", random_state=0)
classifier.fit(X_train, Y_train)

predicted_y = classifier.predict(X_test)
predicted_y

for x in range(len(predicted_y)):
    if (predicted_y[x] != 'no'):
        print(x, end="\t")
