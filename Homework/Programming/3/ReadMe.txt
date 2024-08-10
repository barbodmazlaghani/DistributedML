Computational Assignment 3 Overview
This directory contains solutions to the following tasks focusing on distributed computing using Apache Spark and PySpark:

Word Count and Text Analysis using Spark RDDs (30 points):

Part (a): Load a dataset of news articles, count the total number of articles and words, and print the first 10 words.
Part (b): Convert all words to lowercase, count the frequency of each word, and sort them to print the top 10 most frequent words.
Part (c): Remove punctuation from the words and recalculate the top 10 most frequent words.
Part (d): Find the first letter with the most words starting with it and count the occurrences.
Part (e): Document the code and results for all parts in the report.
TF-IDF Implementation using Spark RDDs (35 points + 5 points for explanation):

Implement the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm from scratch using Spark RDDs on a dataset where each line represents a document.
Calculate and identify the documents most relevant to the words "market", "japan", and "gas".
Provide a detailed explanation of the implementation and the results.
Machine Learning with PySpark ML (30 points + 5 points for execution time):

Part (a): Load a heart disease dataset from HDFS and perform basic data exploration by calculating statistics like mean, variance, minimum, and maximum for specific columns.
Part (b): Split the dataset into training (85%) and testing (15%) sets.
Part (c): Use the PySpark ML pipeline to train Logistic Regression and Random Forest models on the dataset.
Part (d): Evaluate the models using accuracy, precision, recall, and F1-score on the test data.
Part (e): Report the execution time for training each model and summarize the results in the report.