# Deep Learning Project (ACSAI - class 2020/2021)
This repository contains the source code developed for the optional project for the "Deep Learning" course, held by Professor Tony Chi Thong Huynh.

## Authors 
* Benjamin Barda
* Francesco Danese
* Alessandro vecchi

## Introduction
While Data is getting generated at an ever increasing speed and can be retrieved with not much effort, making this data usefull and usable is still not a trivial task at all. One of the main bottlenecks is the need of vast amount of labeled data, which, for a high quality dataset, needs to be manually labeled with care, a process that requires a lot of man-hours to complete. This is the foundamental problem that researchers in the field of Active Learning are trying to answer. **How can we get competitive performance with less data??**. 

Many approaches are possible and we invite to refer to [this repository](https://github.com/baifanxxx/awesome-active-learning) for a comprehensive list of publications on the subject.

The Core idea of Active Learning is to get more out of the human in the loop. 

**But how can we do this?** 

Clearly is not possible to force someone to annotate faster, but what we can do is make sure that every sample that he labels have as much impact as possible. 

<p align="center">
  <img src="https://github.com/Benjamin-Barda/Active-Self-Learning/blob/main/docs/AL-loop.png" alt="Active Learning Loop"/>
</p>

The process is relatively simple from an overview perspective. 

Starting from an existing dataset of unlabeled data we select via an **acquisition function** the samples to be presented to the **oracle** (a human expert for example) for he to annotate them and add them to the labeled dataset. We then train the model on the labeled data. We repeat this process untill exhausting the labeling budget (time, money, ecc..).

## Active Learning: Not only one way

There are three main approaches to the problem : 
* Stream based selective sampling
* Pool-Based sampling
* Membership query synthesis

In this project we investigare Pool-Based sampling but a ìn overview of the various methods [this article](https://www.datarobot.com/blog/active-learning-machine-learning/) might be a good starting point.

## Our Approach

Our Approach can be divided in two main steps. 

* Pretrain
* Active learning loop

### Pretrain

This is a crucial part of the method proposed, in fact the quality of the features learned during this phase have a great impact on the final performnace, so we put great consideration in selecting the appropiate pretext task. 

The one thing that remained constant during all of our experiments were was the choice of the architechture. We decided to use a Residual Network for two reasons:

1. Low number of learnable parameters
2. Proved history of great performance on CIFAR10, the dataset we chose for this project.

During the first phase of the project we considered using the non constrastive task described in [SimSiam](https://arxiv.org/abs/2011.10566). Even though the quality of the features were excellent we faced the problem of limited resources in terms of computing and time, and since to have an effective pretrain we would need to run at least 800 epochs we decided to discard this option. 

We then resorted to a simpler task, more specifically rotation prediction, as described in the [RotNet paper](https://arxiv.org/abs/1803.07728). 
The task is pretty straight forward. Given an unlabeled image, we rotate it by 90, 180 and 270 degreees, associating to each rotation a label (0 for no rotation, 1 for 90deg rotation, ecc). We then train a classifier over the rotated images. The idea is that the network is forced to focus on important features of the image in order to recognize the rotation thus learning usefull features rappresentaions. The big advantage over the SimSiam approach is the realative low number of epochs it needs. In fact around 100 epochs are needed to have an effective pretrain, allowing us to complete the training in a single colab session in less than 2 hours.





