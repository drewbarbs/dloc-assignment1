import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j, :] += X[:, i] # add gradient from scores[j]
        dW[y[i], :] -= X[:, i] # add gradient from -correct_class_score = -scores[y[i]]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_examples = X.shape[1]
  num_classes = W.shape[0]
  col_rng = np.arange(num_examples)

  scores = W.dot(X)  #1
  raw_loss = (scores - scores[y, col_rng]) + 1 #2
  raw_loss[y, col_rng] = 0.0 #3
  indiv_loss = np.maximum(raw_loss, 0.) #4
  data_loss = np.sum(indiv_loss)/num_examples #5
  reg_loss = 0.5 * reg * np.sum(W**2)
  loss = data_loss + reg_loss

  # compute gradient

  d_indiv_loss = np.ones(raw_loss.shape) #5 (divide by num_examples at end)
  d_raw_loss = d_indiv_loss; d_raw_loss[raw_loss < 0] = 0.0 #4
  d_scores = d_raw_loss #2 (positive scores term in #2)
  # for entries of "scores" array corresponding to the correct class,
  # first set d_scores[entries] = 0 (from #3),
  # then subtract 1 for each class for which hingle loss was incurred (negative scores term in #2)
  # combine these two  operations to get the next line
  d_scores[y, col_rng] = -(raw_loss > 0).sum(axis=0) #3,#2
  dW = d_scores.dot(X.T) #1
  dW /= num_examples

  dW += reg*W

  return loss, dW
