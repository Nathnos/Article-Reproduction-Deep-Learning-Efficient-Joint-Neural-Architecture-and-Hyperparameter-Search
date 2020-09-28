#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import argparse
import json


def predict(image_location, classifier):
    img = load_img(image_location, target_size=INPUT_SIZE)
    y = img_to_array(img)
    y = np.expand_dims(y, axis=0)
    pred = classifier.predict(y)
    if pred > 0.5:
        print("Cat")
    else:
        print("Dog")


def print_score(classifier, test_set):
    score = classifier.evaluate(test_set, verbose=0)
    print(f"Test loss: {score[0]:.3f} / Test accuracy: {score[1]:.3f}")


def init_parser():
    parser = argparse.ArgumentParser(description="CNN for Dogs/Cats")
    parser.add_argument(
        "--predict", "-p", help="An image for the CNNto predict"
    )
    return parser.parse_args()


def get_best_params(MODEL_NB):
    params = []
    file_name = f"tests/best_hyperp.{MODEL_NB}"
    with open(file_name, 'r') as file:
        params = json.load(file)
        for p in params:
            print(p)
        #
        # lines = file.readlines()
        # for l in lines:
        #     params.append(float(l))
    # hyperparameters = (
    #     params[5],
    #     params[7],
    #     params[6],
    #     params[9],
    #     params[3],
    #     int(params[1]),
    #     int(params[2]),
    #     int(params[8]),
    #     int(params[4]),
    # )
    # batch_size = int(params[0])
    # return hyperparameters, batch_size
    return None, None
