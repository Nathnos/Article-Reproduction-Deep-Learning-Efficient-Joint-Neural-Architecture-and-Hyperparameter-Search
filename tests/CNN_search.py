#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential
from bayes_opt import BayesianOptimization

from CNN_setup import setup, train, get_sets, EPOCHS, TEST_SET_SIZE

# Hyperparameters bounds
pbounds = {
    "lr": (1e-4, 0.1),
    "momentum": (0.7, 0.95),
    "lr_decay": (0.7, 0.99),
    "weight_decay": (1e-4, 0.1),
    "dropout_rate": (0.1, 0.5),
    "batch_size": (16, 64),
    "conv_layers": (2, 7),
    "dense_layers": (2, 7),
    "nb_filters": (16, 64),
    "kernel_size": (2, 5),
}


def fit_with(
    lr,
    momentum,
    lr_decay,
    weight_decay,
    dropout_rate,
    batch_size,
    conv_layers,
    dense_layers,
    nb_filters,
    kernel_size,
):
    hyperparameters = (
        lr,
        momentum,
        lr_decay,
        weight_decay,
        dropout_rate,
        int(conv_layers),
        int(dense_layers),
        int(nb_filters),
        int(kernel_size),
    )
    batch_size = int(batch_size)
    print("JEUIEIEAUI EAUIETUAIEUAIEAUIEAU", hyperparameters, batch_size)
    classifier = Sequential()
    training_set, test_set = get_sets(batch_size)
    setup(classifier, hyperparameters)
    train(classifier, training_set, batch_size)
    score = classifier.evaluate(
        test_set, steps=TEST_SET_SIZE / batch_size, verbose=0
    )
    return score[1]  # Accuracy


def show_results(bayes_optimizer):
    for i, res in enumerate(bayes_optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(bayes_optimizer.max)


def searching():
    bayes_optimizer = BayesianOptimization(
        f=fit_with, pbounds=pbounds, verbose=2, random_state=1
    )
    bayes_optimizer.maximize(init_points=5, n_iter=10)
    show_results(bayes_optimizer)
