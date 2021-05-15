# [reptile](https://blog.openai.com/reptile/)
implementation of openai's [reptile](https://arxiv.org/abs/1803.02999), a meta-learning algorithm. it works by repeatedly sampling a task T from distribution p(T), performing stochastic gradient descent, and updating global parameters based on the difference between updated and global parameters. 

## what is meta-learning?
meta-learning seeks to initialize models such that, when presented with a new task, it can learn efficiently (from a few examples) and quickly (with a few parameter updates).

## example
we fit a model on sine curves. each curve is unique, as they are randomly initialized with unique amplitude and frequency. for each task, for each task, 50 examples are used for training. 

post training, the model can efficiently learn a new task with a handful of examples, as compared to a randomly initialized network.