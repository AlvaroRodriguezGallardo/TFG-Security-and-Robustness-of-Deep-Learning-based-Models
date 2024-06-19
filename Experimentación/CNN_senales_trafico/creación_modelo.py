import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator


# convertir imagen a escala de gris
def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# ecualizar histograma
def equalize(image):
    image = cv2.equalizeHist(image)
    return image


# preproceso de imagen antes de entrenar el modelo.
def preprocessing(image):
    image = grayscale(image)
    image = equalize(image)
    image = image/255   # normalizar la imagen entre 0 y 1 en vez de 0 y 255
    return image


# unpickle archivos que vienen serializados, mejor forma de almacenar/compartir datos comprimidos
with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

# almacenar en variables el dataset de entrenamiento, validación y test en X
# de igual manera, las etiquetas de cada imagen en y
X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# guardar en variable el diccionario con los nombres de las señales para cada número en el dataset
data = pd.read_csv("german-traffic-signs/signnames.csv")

# con map, ejecutar función preprocessing a cada imagen dentro de los array
X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

# hay que darle profundidad a los datos, ya que el kernel de las capas convolucionales trabajan con profundidad
X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)

# generar nuevos datos de entrenamiento en el dataset modificando morfológicamente los ya existentes
datagen = ImageDataGenerator(width_shift_range=0.1,   # rango en el ancho de la imagen
                             height_shift_range=0.1,    # rango en el ancho de la imagen
                             zoom_range=0.2,    # rango de zoom
                             shear_range=0.1,   # grados de transformación de orte
                             rotation_range=10)  # grados de rotación
# le pasamos al generador los datos de entrenamiento para que recolecte estadística para su transformación
datagen.fit(X_train)

# el generador es un iterador, retorna lotes de imagenes cuando es llamado
batches = datagen.flow(X_train, y_train, batch_size=20)  # batchsize es el n° de imágenes por lote
# al ser un iterador, con "next" retornamos el siguiente valor de la interación.
# guardamos el lote generado en nuevas variables
X_batch, y_batch = next(batches)

# Hacer "one hot encoding" a las etiquetas de los datos para la clasificación multiclase
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)

# Ya preprocesado el dataset, se crea y entrena el modelo a partir de estos datos
num_classes = 43    # nº de clases de datos


# función con la definición de las capas que formará el modelo
def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation="relu"))
    model.add(Conv2D(60, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    #compilar modelo
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# instanciar un modelo de la estructura diseñada en la variable "model"
model = modified_model()
# imprimir listado de características del modelo
print(model.summary())

# finalmente ajustamos el modelo con un entrenamiento a partir de datos de entrenamiento, de validación y sus etiquetas
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch=2000, epochs=10,
                              validation_data=(X_val, y_val), shuffle=1, verbose=1)

# mostrar perdidas del entrenamiento y la validación por cada epoch
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training", "validation"])
plt.title("loss")
plt.xlabel("Epochs")
plt.show()

# mostrar exactitud del entrenamiento y la validación por cada epoch
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["training", "validation"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.show()

# Evaluar el modelo con los datos reservados para el test
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score: ", score[0])
print("Accuracy: ", score[1])

# exportar el modelo para futuras predicciones
#model.save("modelo.h5")