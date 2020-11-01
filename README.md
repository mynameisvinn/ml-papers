# whats dis
a deep dive into one shot learning with the infamous [omniglot](https://github.com/brendenlake/omniglot) dataset.

## omniglot dataset
the omniglot dataset consists of 50 alphabets. 

each alphabet is a collection of characters. for example, the english alphabet has 26 characters. 

each character has 20 examples - eg there are 20 images of the character `a`, 20 images of the character `b`, and so on.

## what is the "one shot learning" problem with omniglot?
given (a) the image of a character from an never-seen-before alphabet (eg perhaps the character `x` from the english alphabet, which youve never seen before) and (b) other images from 19 other never-seen-before alphabets (perhaps russian or chinese), can you match the character `x` to its originating alphabet?

this task is considered “one-shot learning” because you have one shot to correctly classify the letter, despite never seeing that particular alphabet before.

## siamese neural nets to the rescue
one of the most popular approaches to the one-shot learning problem is [siamese neural networks for one shot image recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), presented at iclr 2015.

(there are several other approaches, including [openai's reptile](https://arxiv.org/abs/1803.02999)).

## deep dive into siamese neural nets
* preparing the [training data](https://github.com/mynameisvinn/omniglot/blob/master/preparing_data.md)
* preparing the [test data](https://github.com/mynameisvinn/omniglot/blob/master/preparing_testdata.md)
* [computing loss](https://github.com/mynameisvinn/paper-omniglot/blob/master/loss.md)
* [evaluating](https://github.com/mynameisvinn/omniglot/blob/master/evaluation.md) performance