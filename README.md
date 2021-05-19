# Generalized Learning Vector Quantization For Classification in Randomized Neural Networks And Hyperdimensional Computing

By [Cameron Diao](https://github.com/CameronDiao), [Denis Kleyko](https://github.com/denkle), Jan M. Rabaey, and Bruno Olshausen

We provide the official implementation of "Generalized Learning Vector Quantization For Classification in Randomized Neural Networks And Hyperdimensional Computing".

Our implementation is primarily in Python and includes the following model comparisons (note that RVFL stands for Random Vector Functional Link):

> **Regularized Least Squares (RLS) Classifier vs. Generalized Learning Vector Quantization (GLVQ) Classifier**

> **Conventional RVFL with RLS Classifier vs. Conventional RVFL with GLVQ Classifier**

> **intRVFL with RLS Classifier vs. intRVFL with GLVQ Classifier**

> **intRVFL with RLS Classifier vs. GLVQ-RVFL**
 
> **GLVQ-RVFL vs. Kernel GLVQ (KGLVQ) Classifier**

## Introduction

We proposed a form of RVFL network, called GLVQ-RVFL, that is significantly more computationally efficient than the conventional RVFL.
We devised GLVQ-RVFL by replacing the least-squares classifier of the conventional RVFL with a GLVQ classifier.

GLVQ-RVFL inherits part of its network architecture from intRVFL, an RVFL network proposed by [Kleyko et. al](https://ieeexplore.ieee.org/document/9174774).
Specifically, GLVQ-RVFL inherits the density-encoding and hidden layers of intRVFL.
We then train a GLVQ classifier on the network's hidden layer activation values to issue predictions in the output layer.

We tested GLVQ-RVFL on a collection of 121 real-world classification datasets from the UCI Machine Learning Repository.
We found that GLVQ-RVFL achieved similar accuracy to intRVFL with 21% of the computational cost (in flops).
Without limiting the number of training iterations, we found that the average accuracy of GLVQ-RVFL was 0.82 while the average accuracy of intRVFL was 0.80.