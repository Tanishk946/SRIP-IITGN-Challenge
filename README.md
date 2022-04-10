# SRIP IITGN 2022 - Machine Learning - Prof. Nipun Batra
This repository is about the problems given by professor Nipun Batra for the qualification of Summer Research Internship at IIT Gandhinagar.

I was able to solve only question 3, but I solved it completely with all the specifications mentioned in the problem such as 80:20 training and testing data split, using the pytree concept, plotting loss v/s iterations curve, optimizing the hyper-parameters using only jax.

**Difficulties faced:** The main difficulty i faced was splitting the dataset after studying lot of documentation on pytorch loaders, I was able to split the data.

# Implement two hidden layers neural network classifier from scratch in JAX 
**Abstract:**
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centred in a fixed-size image. The author combined the training and testing dataset in order to split the data into 56000 (80%) training examples and 14000(20%) testing examples. Then, the author applied 2-layered neural network architecture coded from scratch using JAX library and hyper-tuned (changed) parameters. The architecture is performing with an accuracy of 80% approximately on training data and with a 20% accuracy on test data. The accuracy was not so great due to the random shuffle in the data. Various classification evaluation metrics like log loss, accuracy was applied.

**1.	Introduction:**

**Neural Network:**

A neural network is a series of algorithms that endeavours to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature.
**JAX:**

JAX is an automatic differentiation (AD) toolbox developed by a group of people at Google Brain and the open-source community. It aims to bring differentiable programming in NumPy-style onto TPUs. On the highest level JAX combines the previous projects XLA & Autograd to accelerate linear algebra-based projects.
XLA optimizes memory bandwidth by “fusing” operations and reduces the amount of returned intermediate computations. In practice this can help to significantly speed up things. Autograd provides automatic differentiation support for large parts of standard Python features. AD resembles the backbone of optimization in Deep Learning. It simplifies the derivative expression of a compositional function at every possible point in time. JAX supports AD for standard NumPy functions as well as loops which transform numerical variables.

**2.	Classification Results:**
