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

In this project we investigare Pool-Based sampling. 
