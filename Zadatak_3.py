import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]), plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype("float32") / 255.0
X_test_n = X_test.astype("float32") / 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype="uint8")
y_test = to_categorical(y_test, dtype="uint8")

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")
)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation="relu"))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    keras.callbacks.TensorBoard(log_dir="logs/cnn_dropout", update_freq=100),
]

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train_n,
    y_train,
    epochs=20,
    batch_size=32,
    callbacks=my_callbacks,
    validation_split=0.1,
)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f"Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}")

#odgovor za 4.zadatak
#1. 
#Korištenjem velike serije model brže konvergira, ali performanse mogu biti lošije dok korištenjem velike serije dobiva se suprotni učinak
#2.
#opet govorimo o problemu konvergencije(manja sporije konvergira i suprotno), takoder opet možemo govoriti o performansama stoga je najbolje naći zlatnu sredinu
#3.
#izbacivanjem određenih slojeva možemo poremetit mrežu tj. možemo dobivati puno lošije rezultate i mreža neće imati dobru sposobnost klasifikacije podataka
#4.
#smanjenjem skupa smanjit ćemo vrijeme učenja, ali ono što je loše jest to da vjerojatno nećemo idmati dovoljno podataka za pravilno učenje kako bi model bio spreman za nadolazeće podatke koje treba klasificirati
