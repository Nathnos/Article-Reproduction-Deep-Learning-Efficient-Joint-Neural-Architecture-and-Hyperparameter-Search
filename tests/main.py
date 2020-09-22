#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential, load_model, save_model
import argparse
import importlib

#
# from CNN_setup import setup, get_sets, train
# from tools import predict, print_score, init_parser
from CNN_search import searching

# Global settings
NEW_MODEL = True  # If False, loads a model
MORE_TRAINING = True  # In case NEW_MODEL is False
MODEL_NB = 2


def main():
    searching()
    # #Only launch stuff
    # args = init_parser()
    # classifier = Sequential()
    # model_name = f'tests/model{MODEL_NB}.h5'
    # print(model_name)
    # training_set, test_set = get_sets()
    # def launch_training(): #Closure
    #     train(classifier, training_set)
    #     save_model(classifier, model_name)
    # if NEW_MODEL :
    #     setup(classifier)
    #     launch_training()
    # else:
    #     classifier = load_model(model_name)
    #     if MORE_TRAINING:
    #         launch_training()
    # print_score(classifier, test_set)
    # if args.predict is not None :
    #     predict(args.predict, classifier)


if __name__ == "__main__":
    main()