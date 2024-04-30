from PIL import Image
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import models

model = models.load_model('LV8_model.keras')

for i in range(1, 4):
    image_path = f'image{i}.png'
    image = Image.open(image_path)

    plt.imshow(image)
    plt.title('Originalna slika')
    plt.show() 

    plt.imshow(image)
    plt.show()

    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255
    image_array = np.expand_dims(image_array, axis=0)  

    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)

    print(f"Predviđena oznaka slike '{image_path}': {predicted_label}")

