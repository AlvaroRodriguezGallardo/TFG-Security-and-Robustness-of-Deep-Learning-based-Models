# Clasificación de señales de tráfico con OpenCV en Python

Diseño de un clasificador con un modelo de redes neuronales convolucionales. La estructura utilizada se basa en LeNet, diseñada para la clasificación de dígitos escritos a manos como la base de datos MNIST, pero con modificaciones se puede obtener un simple modelo para esta aplicación de señales de tráfico. 

![alt text](https://i.gyazo.com/c8d403c63051e742285049420844f486.png)


En la creación del modelo, se ha utilizado la librería de Python “Keras”, que facilita en gran medida el diseño de estructuras de modelos de Machine Learning sin entrar en las matemáticas en las cuales se basa.
Como base de datos para la obtención del modelo, se usado una recopilación de señales alemanas que se encuentran ya clasificadas con sus etiquetas correspondientes.

https://bitbucket.org/jadslim/german-traffic-signs

![alt text](https://i.gyazo.com/ab4c78ee56e65e1117d2217910209c9a.png)


## Librerías:
* **Keras:** https://keras.io/
* **Tensorflow:** https://www.tensorflow.org/
* **Numpy** http://www.numpy.org/

## Capturas del proceso:
*Preprocesado de imágenes y generación de nuevas.*


&nbsp;
![alt text](https://i.gyazo.com/d4b4aaec8ea4d41cef1e83de54ae9e19.png)

&nbsp;
*Entrenamiento de la base de datos.*
![alt text](https://i.gyazo.com/a24b3f02aef681ca2da1d7f6e130ee00.png)


&nbsp;

*Test de clasificación.*


&nbsp;
![alt text](https://i.gyazo.com/e3a3af9f3c09683870a66f8bb8f2c6ce.png)
![alt text](https://i.gyazo.com/f07d0e46a2d8bf6165352c9a983189cd.jpg)
&nbsp;
