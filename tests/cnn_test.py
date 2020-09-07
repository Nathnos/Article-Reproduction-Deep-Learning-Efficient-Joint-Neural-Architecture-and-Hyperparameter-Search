#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = (64, 64)
NEW_MODEL = True

def setup(classifier) :
    #Add layers
    classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
        input_shape=[*INPUT_SIZE, 3], activation="relu"))

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


def train(classifier):
    # Augmentation d'images
    train_datagen = ImageDataGenerator(
            rescale=1./255, #Valeur pixel entre 0 et 1
            shear_range=0.2, #Transvection
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255) #Tout sur la même échelle !
    training_set = train_datagen.flow_from_directory(
            'data/train',
            target_size=INPUT_SIZE,
            batch_size=32,
            class_mode='binary')
    test_set = test_datagen.flow_from_directory(
            'data/test',
            target_size=INPUT_SIZE,
            batch_size=32,
            class_mode='binary')

    #Trainging itself
    classifier.fit(
            training_set,
            steps_per_epoch=625, #625*32 = 20k
            epochs=20,
            validation_data=test_set,
            validation_steps=157) #156*32 ~= 5k

def main():
    classifier = Sequential()
    setup(classifier)
    if NEW_MODEL :
        train(classifier)
        classifier.save('tests/model0')
    else:
        classifier.load('tests/model0')

if __name__ == "__main__":
    main()
