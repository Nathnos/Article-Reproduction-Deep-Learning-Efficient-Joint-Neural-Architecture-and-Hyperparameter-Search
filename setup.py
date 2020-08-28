#! /usr/bin/env python3
# coding: utf-8

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Article Reproduction : Deep Learning Efficient Joint Neural-"
        + "Architecture and Hyperparameter Search",
    version="0.0.1",
    author="Lorenzo VILLARD",
    author_email="villard.lorenzo.pro@protonmail.com",
    description="Reproduction of a deep learing article",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nathnos/Article-Reproduction-Deep-Learning-"
        + "Efficient-Joint-Neural-Architecture-and-Hyperparameter-Search",
    license="GNUv3",
    python_requires='>=3.6',
)
