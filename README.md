# SRIP IITGN 2022 - Machine Learning - Prof. Nipun Batra
This repository is about the problems given by professor Nipun Batra for the qualification of Summer Research Internship at IIT Gandhinagar.

I was able to solve only question 3, but I solved it completely with all the specifications mentioned in the problem such as 80:20 training and testing data split, using the pytree concept, plotting loss v/s iterations curve, optimizing the hyper-parameters using only jax.

**Difficulties faced:** The main difficulty i faced was splitting the dataset after studying lot of documentation on pytorch loaders, I was able to split the data [3]. Then, the next problem i faced was getting comfartable with the jax libraries, the documentation available on the official jax website helped me a lot to overcome this problem [5]. 

# Implement two hidden layers neural network classifier from scratch in JAX 
**Abstract:**
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centred in a fixed-size image. The images consists of single digits only i.e., 0-9. The author combined the training and testing dataset in order to split the data into 56000 (80%) training examples and 14000(20%) testing examples. Then, the author applied 2-layered neural network architecture coded from scratch using JAX library and hyper-tuned (changed) parameters. The architecture is performing with an accuracy of 79% approximately on training data and with a 19.4% accuracy on test data. The accuracy was not so great due to the random shuffle in the data. Various classification evaluation metrics like log loss, accuracy was applied.

**1.	Introduction:**

**Neural Network:**

A neural network is a series of algorithms that endeavours to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature.

**JAX:**

JAX is an automatic differentiation (AD) toolbox developed by a group of people at Google Brain and the open-source community. It aims to bring differentiable programming in NumPy-style onto TPUs. On the highest level JAX combines the previous projects XLA & Autograd to accelerate linear algebra-based projects.
XLA optimizes memory bandwidth by “fusing” operations and reduces the amount of returned intermediate computations. In practice this can help to significantly speed up things. Autograd provides automatic differentiation support for large parts of standard Python features. AD resembles the backbone of optimization in Deep Learning. It simplifies the derivative expression of a compositional function at every possible point in time. JAX supports AD for standard NumPy functions as well as loops which transform numerical variables.

**2.	Classification Results:**

As the dataset contains 70000 records which is very huge, I had considered applying batch gradient descent and the results with respect to parameters taken are mentioned in the below figures.

Each neural network model is trained for 10 epochs. 

**Batch:100 Layer architecture:[784, 256, 256, 10]**
![100 btch 256 256](https://user-images.githubusercontent.com/53971916/162626528-9d74d235-b830-4234-af65-45cf8d4f7945.png)

**Batch:100 Layer architecture:[784, 512, 512, 10]**
![100 btch 512 512](https://user-images.githubusercontent.com/53971916/162625809-d6bd1312-d24b-40aa-ad25-7fc7c96a2c97.png)

**Batch:500 Layer architecture:[784, 256, 256, 10]**
![500 bch 256 256](https://user-images.githubusercontent.com/53971916/162627081-5ee70305-7e36-49a3-aeaa-4d056a91fc5d.png)

**Batch:500 Layer architecture:[784, 512, 512, 10]**
![500 bch 512](https://user-images.githubusercontent.com/53971916/162627303-0a20d1ee-c798-47bb-9c6f-d3a0ee350c89.png)

**Batch:1000 Layer architecture:[784, 256, 256, 10]**
![1000 bch](https://user-images.githubusercontent.com/53971916/162628066-a14b37ab-206d-41d0-9d8a-b5dd7311b4be.png)

**Batch:1000 Layer architecture:[784, 512, 512, 10]**
![1000 bch 512](https://user-images.githubusercontent.com/53971916/162627754-65602d1b-9448-4542-971e-39994e9093da.png)

**3.	Loss v/s Iteration plots:**

The loss v/s iteration plots were drawn for 10 epochs and with the below mentioned hyper-parameters.

**Batch:100 Layer architecture:[784, 256, 256, 10]**
![100 batch 256 256](https://user-images.githubusercontent.com/53971916/162627515-48cb88f5-37fe-4cec-8012-c7a9a3893a19.png)

**Batch:100 Layer architecture:[784, 512, 512, 10]**
![100 batch 512 512](https://user-images.githubusercontent.com/53971916/162627531-a61a366b-88fd-4a02-b8b7-c0eb1c6e0bae.png)

**Batch:500 Layer architecture:[784, 256, 256, 10]**
![500 batch 256 256](https://user-images.githubusercontent.com/53971916/162627542-c489f830-513e-4769-94ec-ee3e02ced645.png)

**Batch:500 Layer architecture:[784, 512, 512, 10]**
![500 batch 512 512](https://user-images.githubusercontent.com/53971916/162627555-a3287b81-6dac-4d46-bb0c-e0c21972ebd2.png)

**Batch:1000 Layer architecture:[784, 256, 256, 10]**
![1000 batch](https://user-images.githubusercontent.com/53971916/162628059-17b42a1d-c29d-49f8-be61-947ced3d9d13.png)

**Batch:1000 Layer architecture:[784, 512, 512, 10]**
![1000 batch 512 512](https://user-images.githubusercontent.com/53971916/162627771-cd3bb450-6b28-41b7-94ab-fb1e32082604.png)

**4.	Conclusion and Future Scope**

The author had tested various neural network architectures on the mnist dataset with 80:20 train and test split ration. Almost, all the architectures are working similarly with a training accuracy of 79% and testing accuracy of 19.4% approximately. In the future, the author will implement the pre-trained convolutional neural network architectures like efficientnet, teacher-student model for improving the performance.

**References**

[1].https://roberttlange.github.io/posts/2020/03/blog-post-10/

[2].https://www.youtube.com/watch?v=6_PqUPxRmjY

[3].https://pytorch.org/vision/stable/datasets.html

[4].http://yann.lecun.com/exdb/mnist/

[5].https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
