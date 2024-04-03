import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# Brojanje primjera za svaku klasu u skupu podataka za učenje
train_classes, train_counts = np.unique(y_train, return_counts=True)

# Brojanje primjera za svaku klasu u skupu podataka za testiranje
test_classes, test_counts = np.unique(y_test, return_counts=True)

# Prikaz rezultata pomoću stupčastog dijagrama
plt.figure(figsize=(10, 5))
plt.bar(train_classes, train_counts, color='blue', alpha=0.5, label='Train Data')
plt.bar(test_classes, test_counts, color='red', alpha=0.5, label='Test Data')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances for Each Class')
plt.legend()
plt.show()

Log_RegressionModel = LogisticRegression()
Log_RegressionModel.fit(X_train,y_train)

# # Pronalaženje parametara modela
coefs=Log_RegressionModel.coef_
intercepts=Log_RegressionModel.intercept_
for i, label in labels.items():
    print(f"Koeficijenti za klasu '{label}': {coefs[i]}")
for i, label in labels.items():
    print(f"Odsjecak za klasu '{label}': {intercepts[i]}")

# Poziv funkcije plot_decision_regions s podacima za učenje i izgrađenim modelom logističke regresije
plot_decision_regions(X_train, y_train.ravel(), Log_RegressionModel)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Decision Regions - Logistic Regression')
plt.show()


# Predviđanje klasa za testni skup
y_pred = Log_RegressionModel.predict(X_test)

# Izračun matrice zabune
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(conf_matrix)

# Izračun točnosti
accuracy = accuracy_score(y_test, y_pred)
print("Tocnost:", accuracy)

# Prikaz classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[labels[i] for i in range(3)]))
