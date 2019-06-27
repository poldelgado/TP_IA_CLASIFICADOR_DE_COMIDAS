import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #Capas donde haremos las convoluciones y maxpooling
from tensorflow.python.keras import backend as K

K.clear_session() #Eliminamos cualquier sesion de keras que este corriendo en background

data_entrenamiento ='./data/training' #Carpeta donde tenemos las imagenes para entrenar
data_validacion='./data/validation' #Carpeta donde hacemos la validacion

##Parametros

epocas = 25
altura, longitud = 100, 100
batch_size = 50 #nro de imagenes q enviamos a procesar en cada paso
pasos = 1250 #nro de veces q se procesara la info en cada epoca
pasos_validacion = 200
filtrosConv1 = 32 #Despues de 1era conv la imagen tendra una prof de 32
filtrosConv2 = 64 #Idem anterior xo 64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool = (2,2) #tamaño de filtro en el max_pooling
clases = 11 #cambiar al nro de clases q tengamos
lr = 0.005 # learning rate

## pre procesamiento de imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

validacion_datagen = ImageDataGenerator(
    rescale = 1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

#Crear la red CNN

cnn = Sequential() #la red q generaremos es secuencial (varias capas apiladas)
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape = (altura, longitud, 3), activation = 'relu'))

cnn.add(MaxPooling2D(pool_size=tamano_pool)) #Capa de max_pooling

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten()) #La imagen q en este momento es profunda pero pequeña la aplanamos
cnn.add(Dense(256, activation='relu')) #Luego de aplanar la info la enviamos a una capa densa
cnn.add(Dropout(0.5)) #A la capa densa durante el entrenamiento apagamos el 50% de las neuronas en cada paso, esto se realiza para evitar ajustar, ya que si todo el tiempo las neuronas estan activadas puede que nuestra red neuronal aprenda un camino especifico para especificar por ej perros
cnn.add(Dense(clases, activation='softmax')) #Softmax nos dice el porcentaje de la imagen que puede ser.

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy']) #parametros para optimizar nuestro algoritmo
# cnn.compile(loss='categorical_crossentropy',
#             optimizer=optimizers.Adam(lr=lr),
#             metrics=['accuracy'])

cnn.fit_generator(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)

# cnn.fit_generator(
#     entrenamiento_generador,
#     steps_per_epoch=pasos,
#     epochs=epocas,
#     validation_data=validacion_generador,
#     validation_steps=validation_steps)


# if not os.path(dir):
#     os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')