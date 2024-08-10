Computational Assignment 4 Overview
This directory contains solutions to tasks focused on distributed training and profiling of neural networks using PyTorch:

Distributed Model Training with PyTorch (50 points):

Part (a): Implement a FeedForward neural network model with ReLU activation, BatchNorm layers, and an optimizer of your choice. Train the model to achieve over 80% accuracy on the test data, using a specified dataset converted from float16 to float32.
Part (b): Save a model checkpoint at the end of training using only one process.
Part (c): Train the model using torchrun and slurm in four different configurations:
Single machine with a single core.
Single machine with two cores.
Two machines, each with one core.
Two machines, each with two cores.
Compare and report the accuracy and training time for each configuration.
Profiling Neural Networks with PyTorch Profiler (50 points):

Part (a): Load the trained model and test it on a batch of 100 test samples using the CPU. Profile the execution, focusing on time and memory usage for the BatchNorm, ReLU, and Linear layers. Report and compare the results.
Part (b): Replace the ReLU activation function with Sigmoid, Tanh, and GeLU, and compare the time and memory usage for each function.
Part (c): Compare the time and memory usage of using BatchNorm versus Dropout layers.