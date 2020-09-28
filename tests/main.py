#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential, load_model, save_model
import argparse
import importlib


from CNN_setup import setup, get_sets, train
from tools import predict, print_score, init_parser, get_best_params
from CNN_search import searching

# Global settings
SEARCHING = False
NEW_MODEL = True  # If False, loads a model
MORE_TRAINING = False  # In case NEW_MODEL is False
MODEL_NB = 3


def main():
    hyperparameters, batch_size = get_best_params(MODEL_NB)
    print(hyperparameters, batch_size)
    # searching(MODEL_NB)
    # args = init_parser()
    # model_name = f"tests/model{MODEL_NB}.h5"
    #
    # def launch_training():
    #     train(classifier, training_set, batch_size, train=True)
    #     save_model(classifier, model_name)
    #
    # if SEARCHING:
    #     searching(MODEL_NB)
    #
    # # Then, construct a model with best parameters
    # # TODO open result file, get hyperparameters
    # hyperparameters = (0.01, 0.9, 0.01, 0.01, 0.2, 2, 3, 32, 5)
    # if NEW_MODEL:
    #     classifier = setup(classifier, hyperparameters)
    #     launch_training()
    # else:
    #     classifier = load_model(model_name)
    #     if MORE_TRAINING:
    #         launch_training()
    # print_score(classifier, test_set)
    # if args.predict is not None:
    #     predict(args.predict, classifier)


if __name__ == "__main__":
    main()
