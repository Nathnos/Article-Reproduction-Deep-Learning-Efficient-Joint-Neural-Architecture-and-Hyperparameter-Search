#! /usr/bin/env python3
# coding: utf-8

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import RMSprop
from keras.regularizers import l2
import numpy as np


# Fixed hyperparameters
SEARCHING_EPOCHS = 1  # 100 TODO
TRAINING_EPOCHS = 1  # 500 TODO
UNITS_FACTOR = 2  # Each layer double the number of neurones
MAX_NEURONES = 128
INPUT_SIZE = (258, 258)  # For pictures
STARTING_UNITS = 64  # For Dense layers
CONV_STRIDES = 2
CONV_MAX_POOL_SIZE = (2, 2)
TRAINING_SET_SIZE = 20000
TEST_SET_SIZE = 5000


def setup(classifier, hyperparameters):
    (
        lr,
        momentum,
        lr_decay,
        weight_decay,
        dropout_rate,
        conv_layers,
        dense_layers,
        nb_filters,
        kernel_size,
    ) = hyperparameters
    # First layer, compulsory
    classifier.add(
        Convolution2D(
            filters=nb_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=CONV_STRIDES,
            input_shape=(*INPUT_SIZE, 3),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="valid",
        )
    )
    classifier.add(MaxPooling2D(pool_size=CONV_MAX_POOL_SIZE))
    # Other layers
    for i in range(1, conv_layers):
        classifier.add(
            Convolution2D(
                filters=min(nb_filters * UNITS_FACTOR ** i, MAX_NEURONES),
                kernel_size=(kernel_size, kernel_size),
                strides=CONV_STRIDES,
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
            )
        )
        classifier.add(MaxPooling2D(pool_size=CONV_MAX_POOL_SIZE))

    classifier.add(Dropout(dropout_rate))
    classifier.add(Flatten())

    # Fully Connected Network
    for i in range(dense_layers):
        classifier.add(
            Dense(
                units=min(STARTING_UNITS * UNITS_FACTOR ** i, MAX_NEURONES),
                activation="relu",
                kernel_regularizer=l2(weight_decay),
            )
        )
    # Last layer :
    classifier.add(Dropout(dropout_rate))
    classifier.add(Dense(units=1, activation="sigmoid"))

    # Compile
    rms_opti = RMSprop(lr=lr, momentum=momentum, decay=lr_decay)
    classifier.compile(
        optimizer=rms_opti, loss="binary_crossentropy", metrics=["accuracy"]
    )


def get_sets(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Valeur pixel entre 0 et 1
        shear_range=0.2,  # Transvection
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Même échelle !
    training_set = train_datagen.flow_from_directory(
        "data/train",
        target_size=INPUT_SIZE,
        batch_size=batch_size,
        class_mode="binary",
    )
    test_set = test_datagen.flow_from_directory(
        "data/test",
        target_size=INPUT_SIZE,
        batch_size=batch_size,
        class_mode="binary",
    )
    # Those are (x, y) tuples, x being a numpy array of the image, and y label
    return training_set, test_set


def train(classifier, training_set, batch_size, train=False):
    EPOCHS = TRAINING_EPOCHS if train else SEARCHING_EPOCHS
    classifier.fit(
        training_set,
        # steps_per_epoch=TRAINING_SET_SIZE / batch_size, TODO
        steps_per_epoch=3,
        epochs=EPOCHS,
        verbose=1,
    )
