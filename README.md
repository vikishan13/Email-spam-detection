# Spam Detection using Universal Sentence Encoder and Neural Networks

## Overview

This project demonstrates a spam detection system using the Universal Sentence Encoder (USE) and a simple neural network implemented with TensorFlow. The Universal Sentence Encoder is used to convert textual data into fixed-size embeddings, and a neural network is trained to classify whether a given text is spam or not.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Custom Input Prediction](#custom-input-prediction)
- [License](#license)

## Introduction

Spam detection is a common problem in natural language processing, and this project addresses it by utilizing the Universal Sentence Encoder, a pre-trained model developed by Google, to convert text data into embeddings. These embeddings are then used as input to a neural network that is trained to classify messages as either spam or not spam.

## Project Structure

- **`main.py`**: The main script that loads the USE, constructs the neural network, and trains the model.
- **`utils.py`**: Contains utility functions for data loading, preprocessing, and evaluation.
- **`trained_model/`**: A directory to store the trained model.
- **`data/`**: Placeholder for your dataset (replace with your actual dataset).

## Dependencies

- TensorFlow
- TensorFlow Hub
- scikit-learn

## Install dependencies using:
pip install tensorflow tensorflow-hub scikit-learn

## Model Architecture
The neural network architecture consists of a single dense layer with ReLU activation followed by regularization parameter and a final dense layer with sigmoid activation for binary classification.

## Training
The model is trained using the training set, and the training process involves minimizing the binary crossentropy loss using the Adam optimizer.

**model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, labels_train, epochs=5, batch_size=32, validation_split=0.2)**_

## Evaluation
The model is evaluated on the test set, and accuracy along with a classification report and confusion matrix.

## Custom Input Prediction
You can predict whether a custom input is spam or not using the provided script.
