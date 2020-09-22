#! /usr/bin/env python3
# coding: utf-8

from CNN_setup import setup, train, get_sets
from CNN_setup import BATCH_SIZE, EPOCHS, TEST_SET_SIZE

#Hyperparameters to tune
lr = 0.01 # Learning rate
momentum = 0.9
lr_decay = 0.9
dropout_rate = 0.25
batch_size = 32
conv_layers = 2
dense_layers = 3
nb_filters = 32
kernel_size = 3

#Hyperparameters bounds
pbounds = {'lr' : (1e-4, 0.01),
    'momentum' : (0.7, 0.95),
    'lr_decay' : (0.7, 0.99),
    'dropout_rate' : (0.25),
    'batch_size' : (16, 64),
    'conv_layers' : (2, 7),
    'dense_layers' : (2, 7),
    'nb_filters' : (16, 64),
    'kernel_size' : (2, 5)}

test_set, training_set = get_sets()

def fit_with(classifier, hyperparameters):
    classifier = Sequential()
    setup(classifier, hyperparameters)
    train(classifier, training_set, EPOCHS)
    score = classifier.evaluate(test_set, steps=TEST_SET_SIZE/BATCH_SIZE,
        verbose=0)
    return score[1] #Accuracy

def searching() :
    #TODO : loop
    hyperparameters =  (lr, momentum, lr_decay, dropout_rate, batch_size,
    conv_layers, dense_layers, nb_filters, kernel_size) #TODO bay search
    acc_score = fit_with(classifier, hyperparameters)
    print(acc_score, hyperparameters) #TODO in a file
