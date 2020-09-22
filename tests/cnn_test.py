#! /usr/bin/env python3
# coding: utf-8

from keras.models import Sequential, load_model, save_model
import argparse
import importlib

from CNN_tools import setup, get_sets, train, predict

#Global settings
NEW_MODEL = False #If False, loads a model
MORE_TRAINING = True #In case NEW_MODEL is False
MODEL_NB = 1

def print_score(classifier, test_set):
    score = classifier.evaluate(test_set, verbose=0)
    print(f'Test loss: {score[0]:.3f} / Test accuracy: {score[1]:.3f}')

def init_parser():
    parser = argparse.ArgumentParser(description="CNN for Dogs/Cats")
    parser.add_argument('--predict', '-p',
        help='An image for the CNNto predict')
    return parser.parse_args()


def main():
    args = init_parser()
    classifier = Sequential()
    model_name = f'tests/model{MODEL_NB}.h5'
    print(model_name)
    training_set, test_set = get_sets()
    def launch_training(): #Closure
        train(classifier, training_set)
        save_model(classifier, model_name)
    if NEW_MODEL :
        setup(classifier)
        launch_training()
    else:
        classifier = load_model(model_name)
        if MORE_TRAINING:
            launch_training()
    print_score(classifier, test_set)
    if args.predict is not None :
        predict(args.predict, classifier)

if __name__ == "__main__":
    main()
