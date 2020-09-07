#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


#Initialize CNN
classifier = Sequential()

#Add layers
input_size = (64, 64)
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
    input_shape=[*input_size, 3], activation="relu"))

"""
kernel_size : 3, 5, 7…
filters : puissance de 2, on double à chaque couche
input_shape : on force toutes les images à la même taille
"""

classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(Flatten())

#Fully Connected Network
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))
# classifier.add(Dense(units=2, activation="softmax"))

#Compile
classifier.compile(optimizer="adam", loss="binary_crossentropy",
    metrics=["accuracy"])

# Augmentation d'images
train_datagen = ImageDataGenerator(
        rescale=1./255, #Valeur pixel entre 0 et 1
        shear_range=0.2, #Transvection
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255) #Tout sur la même échelle !
training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=input_size,
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'data/test',
        target_size=input_size,
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=test_set,
        validation_steps=800)



print("oui.")
