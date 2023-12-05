import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

filename = "/Users/noooo/PycharmProjects/pythonProject3/08_pima-indians-diabetes.data.csv"

column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data = pd.read_csv(filename, names = column_names)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(data.groupby('class').size())
print(data.shape)
print(data.describe())

model = DecisionTreeClassifier(max_depth=1000, min_samples_split=60, min_samples_leaf=5)

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, y, cv=kfold)

scatter_matrix(data)
plt.savefig("./results/scatter_plot.png")

print(results.mean())