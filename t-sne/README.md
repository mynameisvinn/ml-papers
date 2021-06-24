# pie-sne
a toy implementation of [t-sne](http://www.cs.toronto.edu/~hinton/absps/tsne.pdf). 

## how dey do dat's
relative distances between points in high dimensional space should be preserved when theyre mapped to lower dim space.

in order to capture this constraint, we'd like to use some similarity/distance measure (which would act as a differentiable loss function that tells us how to arrange points in lower dimensional space). 

a simple, popular and differentiable measure of distance is [kullback leibler](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). since kl divergence works on probability distributions, we need to transform euclidean distances (eg l2) into probability densities. that is, the distance between point A and point B should not be an euclidean distance, but a probability.

we convert euclidean distances between two points by applying a gaussian centered on point A. the euclidean distance between point A and point B is then represented as the probability of seeing point B given a gaussian distribution centered on point A. since the probability mass is centered around point A, this distance-to-probability mapping strongly preserves local neighborhoods instead of global structure.

now that we have relative distances (in both high and low dimensional space) represented as probability distributions, we can find a set of points in low dimensional space that would minimize KL divergence through backprop.