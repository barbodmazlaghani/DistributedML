Computational Assignment 2 Overview
This directory contains solutions to the following tasks focused on distributed training and optimization of neural networks using PyTorch:

Training a Convolutional Neural Network on Fashion MNIST using PyTorch DDP (35 points):

Part (a): Implement and train a CNN on a single GPU to classify the Fashion MNIST dataset using the Adam optimizer.
Part (b): Modify the code to run on two GPUs using Python multiprocessing.
Part (c): Further modify the code to run on two GPUs using the torchrun utility.
Part (d): Compare and analyze the execution time, final model accuracy, and GPU memory usage for each setup.

Impact of Batch Size on Training Speed (15 points):

Investigate the effect of different batch sizes (16, 64, 128) on training speed, final model accuracy, and GPU memory usage.
Plot and analyze the results to understand how batch size impacts these metrics.

Gradient Accumulation Technique (25 points):

Part (a): Modify the code to implement the gradient accumulation technique and explain its purpose.
Part (b): Test the implementation with different step sizes (4, 8, 16) and analyze the effect on training time, model accuracy, and GPU memory usage.
Part (c): Suggest scenarios where gradient accumulation would be beneficial.

Optimizer Comparison with Fixed Batch Size (15 points):

Compare different optimizers (SGD, Adagrad, RMSprop, Adam) in terms of training speed, accuracy, and GPU memory usage, using a fixed batch size of 128.

Backend Communication in PyTorch DDP (10 points):

Compare different backends (nccl, gloo) in PyTorch DDP for their impact on training speed, model accuracy, and GPU memory usage, using the Adam optimizer and a batch size of 128.