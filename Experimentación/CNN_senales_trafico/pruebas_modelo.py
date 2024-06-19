from keras.models import load_model
import cv2
import numpy as np
import pandas as pd


def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def equalize(image):
    image = cv2.equalizeHist(image)
    return image


def preprocessing(image):
    image = grayscale(image)
    image = equalize(image)
    image = image/255   # normalizar la imagen entre 0 y 1 en vez de 0 y 255
    return image


# función para la clasificación
def classification(img):
    # copia para mostrar la imagen con su etiqueta de predicción
    output = img.copy()
    # adaptar resolución a 32x32, la utilizada por el modelo de Keras clasificador
    img = cv2.resize(img, (32, 32))
    # preprocesar imagen a escala de gris y ecualizar
    img = preprocessing(img)
    cv2.imshow("preprocessing", cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA))
    # dar la forma que utiliza Keras en redes convolucionales
    img = img.reshape(1, 32, 32, 1)
    # almacenar la mayor probabilidad de ser una de las 43 clases
    probability = np.amax(model.predict(img)) * 100
    # igualmente, donde se encuentra esa predicción para saber su etiqueta
    class_pre = np.argmax(model.predict(img))
    # leer el csv donde se encuntran las etiquetas
    data = pd.read_csv("german-traffic-signs/signnames.csv")
    # asginar etiqueta según el diccionario
    class_pre = data.SignName[class_pre]
    # etiqueta a imprimir en la imagen de salida
    label = "{}: {:.2f}%".format(class_pre, probability)
    # definir tamaño de la imagen de salida
    output = cv2.resize(output, (500, 500), interpolation=cv2.INTER_AREA)
    # rectangulo detras del texto
    cv2.rectangle(output, (0, 0), (370, 40), (255, 255, 255), -1)
    # impresión de la etiqueta en la imagen
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    # mostrar imagen
    cv2.imshow("result", output)
    # imprimir en el terminal la etiqueta de la imagen predicha
    print(label)
    cv2.waitKey(0)


# cargar el modelo ya generado y exportado
model = load_model("modelo_v3.h5")

# mostrar un conjunto de imagenes con su predición
for i in range(1, 9):
    cv2_image = cv2.imread("test/"+str(i)+".jpg")
    classification(cv2_image)


