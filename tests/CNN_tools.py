#! /usr/bin/env python3
# coding: utf-8

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import RMSprop
import numpy as np

#Global settings (16 hyperparametrs to tune)
EPOCHS = 1 #Fixed
UNITS_FACTOR = 2 #Fixed
#If UNITS_FACTOR = 2, each layer double the number of neurones
LEARNING_RATE = 0.01
BATCH_SIZE = 32
DECAY = 0.9
MOMENTUM = 0.9

#Convolution
INPUT_SIZE = (128, 128) #For pictures
NB_CONV_LAYERS = 2 #Number of layers
CONV_DROPOUT_RATE = 0.25
#Conv + relu
CONV_FILTERS = 32
KERNEL_SIZE = 3
CONV_STRIDES = 2
#Max pool
CONV_MAX_POOL_SIZE = (2, 2)

#Dense
NB_DENSE_LAYERS = 3
DENSE_DROPOUT_RATE = 0.5
STARTING_UNITS = 64

def setup(classifier) :
    #First, compulsory, layer
    classifier.add(Convolution2D(filters=CONV_FILTERS,
        kernel_size=KERNEL_SIZE, strides=CONV_STRIDES,
        input_shape=[*INPUT_SIZE, 3], activation="relu"))
    classifier.add(MaxPooling2D(pool_size=CONV_MAX_POOL_SIZE))
    #Other layers
    for i in range(1, NB_CONV_LAYERS):
        classifier.add(Convolution2D(filters=CONV_FILTERS*UNITS_FACTOR**i,
            kernel_size=KERNEL_SIZE, strides=CONV_STRIDES,
            activation="relu"))
        classifier.add(MaxPooling2D(pool_size=CONV_MAX_POOL_SIZE))
    classifier.add(Dropout(CONV_DROPOUT_RATE))
    classifier.add(Flatten())

    #Fully Connected Network
    for i in range(NB_DENSE_LAYERS):
        classifier.add(Dense(units=(int)(STARTING_UNITS*UNITS_FACTOR**i),
            activation="relu"))
    #Last layer :
    classifier.add(Dropout(DENSE_DROPOUT_RATE))
    classifier.add(Dense(units=1, activation="sigmoid"))

    #Compile
    rms_opti = RMSprop(lr=LEARNING_RATE, momentum=MOMENTUM, decay=DECAY)
    classifier.compile(optimizer=rms_opti, loss="binary_crossentropy",
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
            batch_size=BATCH_SIZE,
            class_mode='binary')
    test_set = test_datagen.flow_from_directory(
            'data/test',
            target_size=INPUT_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary')
    #Those are (x, y) tuples, x being a numpy array of the image, and y label
    return training_set, test_set

def train(classifier, training_set):
    classifier.fit(training_set, steps_per_epoch=20000/BATCH_SIZE,
        epochs=EPOCHS)

def predict(image_location, classifier):
    img = load_img(image_location, target_size=INPUT_SIZE)
    y = img_to_array(img)
    y = np.expand_dims(y, axis=0)
    pred = classifier.predict(y)
    if(pred > 0.5):
        print("Cat")
    else :
        print("Dog")
