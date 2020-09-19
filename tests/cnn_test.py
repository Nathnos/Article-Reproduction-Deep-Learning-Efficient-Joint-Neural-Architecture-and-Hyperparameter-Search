#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential, load_model, save_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = (64, 64) #(128, 128) #For pictures
NEW_MODEL = False #If False, loads a model
#Number of layers
NB_CONV_LAYERS = 2
#Conv + relu
CONV_FILTERS = 32
KERNEL_SZ = 3
CONV_STRIDES = 1
#Max pool
CONV_MAX_POOL_SIZE = (2, 2)
#Dense
NB_DENSE_LAYERS = 1
STARTING_UNITS = 128
UNITS_FACTOR = 2
EPOCHS = 1 #20

def setup(classifier) :
    #First, compulsory, layer
    classifier.add(Convolution2D(filters=CONV_FILTERS, kernel_size=KERNEL_SZ,
        strides=CONV_STRIDES, input_shape=[*INPUT_SIZE, 3], activation="relu"))
    classifier.add(MaxPooling2D(pool_size=CONV_MAX_POOL_SIZE))
    #Other layers
    for i in range(1, NB_CONV_LAYERS):
        classifier.add(Convolution2D(filters=CONV_FILTERS,
            kernel_size=KERNEL_SZ, strides=CONV_STRIDES, activation="relu"))
        classifier.add(MaxPooling2D(pool_size=CONV_MAX_POOL_SIZE))
    classifier.add(Flatten())

    """
    kernel_size : 3, 5, 7…
    filters : puissance de 2, on double à chaque couche
    input_shape : on force toutes les images à la même taille
    """

    #Fully Connected Network
    for i in range(NB_DENSE_LAYERS):
        classifier.add(Dense(units=STARTING_UNITS*UNITS_FACTOR**i,
            activation="relu"))
    #Last layer :
    classifier.add(Dense(units=1, activation="sigmoid"))
    # classifier.add(Dense(units=2, activation="softmax"))

    #Compile
    classifier.compile(optimizer="adam", loss="binary_crossentropy",
        metrics=["accuracy"])

def get_sets():
    train_datagen = ImageDataGenerator(
            rescale=1./255, #Valeur pixel entre 0 et 1
            shear_range=0.2, #Transvection
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255) # Même échelle !
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
    return training_set, test_set

def train(classifier, training_set):
    classifier.fit(
            training_set,
            steps_per_epoch=625, #625*32 = 20k
            epochs=EPOCHS,
            validation_steps=157) #156*32 ~= 5k

def print_score(classifier, test_set):
    score = classifier.evaluate(test_set, verbose=0)
    print(f'Test loss: {score[0]:.3f} / Test accuracy: {score[1]:.3f}')

def main():
    classifier = Sequential()
    training_set, test_set = get_sets()
    if NEW_MODEL :
        setup(classifier)
        train(classifier, training_set)
        save_model(classifier, 'tests/model0.h5')
    else:
        classifier = load_model('tests/model0.h5')
    print_score(classifier, test_set)

if __name__ == "__main__":
    main()
