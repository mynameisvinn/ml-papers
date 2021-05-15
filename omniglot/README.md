# One shot learning with Siamese Networks.
A deep dive into one shot learning with the infamous [Omniglot](https://github.com/brendenlake/omniglot) dataset. 

The omniglot dataset consists of 50 alphabets. Each alphabet is a collection of characters. for example, the english alphabet has 26 characters. Each character has 20 examples - eg there are 20 images of the character `a`, 20 images of the character `b`, and so on. You can get the dataset from [here](https://github.com/brendenlake/omniglot/tree/master/python).

## What is the "one shot learning" problem?
Given (a) an image of a character from an never-seen-before alphabet (eg perhaps the character `x` from the English alphabet, which you've never seen before) and (b) images from 19 other never-seen-before alphabets (perhaps Russian or Chinese), can you match the character `x` to its corresponding alphabet?

This task is considered “one-shot learning” because you have to map the character to its alphabet despite never seeing that particular alphabet before.

## Siamese neural nets to the rescue
One of the most popular approaches to the one-shot learning problem is [siamese neural networks for one shot image recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), presented at iclr 2015.

(There are several other approaches, including [openai's reptile](https://arxiv.org/abs/1803.02999)).

## deep dive into siamese neural nets
* preparing the [training data](https://github.com/mynameisvinn/ml-papers/blob/master/omniglot/preparing_data.md)
* preparing the [test data](https://github.com/mynameisvinn/omniglot/blob/master/preparing_testdata.md)
* [computing loss](https://github.com/mynameisvinn/ml-papers/blob/master/omniglot/loss.md)
* [evaluating](https://github.com/mynameisvinn/omniglot/blob/master/evaluation.md) performance
* [Source](https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a)