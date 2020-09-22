#! /usr/bin/env python3
# coding: utf-8

import numpy as np

def predict(image_location, classifier):
    img = load_img(image_location, target_size=INPUT_SIZE)
    y = img_to_array(img)
    y = np.expand_dims(y, axis=0)
    pred = classifier.predict(y)
    if(pred > 0.5):
        print("Cat")
    else :
        print("Dog")

def print_score(classifier, test_set):
    score = classifier.evaluate(test_set, verbose=0)
    print(f'Test loss: {score[0]:.3f} / Test accuracy: {score[1]:.3f}')

def init_parser():
    parser = argparse.ArgumentParser(description="CNN for Dogs/Cats")
    parser.add_argument('--predict', '-p',
        help='An image for the CNNto predict')
    return parser.parse_args()
