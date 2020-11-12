# lime
a minimal walk through lime

## what is the task?
we have access to a high capacity black box model `f`, capable of fitting nonlinear decision boundaries. 

can we use `f` to learn a linear model `g` within a small region  of our input space, which would help us identify important features?

## data for binary classification
![data](./data.png)

we generate data for binary classification with a nonlinear decision boundary (represented as a sine curve). 

