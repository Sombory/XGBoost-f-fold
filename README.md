# XGBoost-f-fold
Model-for-clasification-with-k-fold

@pam

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('C:/Users/eeliano/Desktop/Codeo R/Codeo/Python/machinelearning-az-master/datasets/Part 10 - Model Selection & Boosting/Section 49 - XGBoost/Churn_Modelling.csv')


dataset.head()

# Divido entre variables explicativas (X) y a predecir (Y)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print(type(X))
X

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [2])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

!pip install xgboos
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.std()
accuracies.mean(


