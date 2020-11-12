# lime
a minimal walkthrough lime.

## what is the task?
we have access to a black box model `f`, capable of fitting nonlinear decision boundaries. 

we want to use `f` to learn a linear model `g` for a small region of interest in our input space `r`, such that `g` could identify important features in `r`.

## what does the data look like?
![data](./data.png)

we generate data for binary classification with a nonlinear decision boundary (in this case, a sine curve). 

