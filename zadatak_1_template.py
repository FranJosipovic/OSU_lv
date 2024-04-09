import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure(figsize=(10,5))
plt.scatter(x=X_train[:,0],y=X_train[:,1],c=y_train,marker='o',cmap="viridis",label="train data")
plt.scatter(x=X_test[:,0],y=X_test[:,1],c=y_test,marker='x',cmap="viridis",label="test data")
plt.xlabel('X1')
plt.ylabel('X2')

Log_RegressionModel = LogisticRegression()
Log_RegressionModel.fit(X_train,y_train)

# Prikaži granicu odluke modela
theta0 = Log_RegressionModel.intercept_
theta1, theta2 = Log_RegressionModel.coef_[0]

# Granica odluke: theta0 + theta1*x1 + theta2*x2 = 0
# Pretvorimo ovu jednačinu u oblik y = mx + b
# gde je y = x2, x = x1, m = -theta1/theta2, b = -theta0/theta
x1_values = np.linspace(min(X_train[:,0]), max(X_train[:,0]), 100)
x2_values = - (theta1/theta2) * x1_values - (theta0/theta2)
plt.plot(x1_values, x2_values, color='red', label='Decision Boundary')
# Klasifikacija testnog skupa
y_pred = Log_RegressionModel.predict(X_test)
plt.scatter(x=X_test[:,0],y=X_test[:,1],c=y_pred,marker='s',label="predict data",cmap='viridis')
plt.legend()
plt.show()

# Priprema podataka za prikaz
correctly_classified = X_test[y_test == y_pred]
misclassified = X_test[y_test != y_pred]

# Prikaz skupa za testiranje
plt.figure(figsize=(10, 5))
plt.scatter(x=correctly_classified[:, 0], y=correctly_classified[:, 1], c='green', marker='o', label='Correctly Classified')
plt.scatter(x=misclassified[:, 0], y=misclassified[:, 1], c='black', marker='x', label='Misclassified')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(conf_matrix)

# Izračun točnosti, preciznosti i odziva
accuracy = accuracy_score(y_test, y_pred)
print("Tocnost:", accuracy)

precision = precision_score(y_test, y_pred)
print("Preciznost:", precision)

recall = recall_score(y_test, y_pred)
print("Odziv:", recall)

f1 = f1_score(y_test,y_pred)
print("F1:", recall)
