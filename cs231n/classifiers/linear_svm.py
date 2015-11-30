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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = W.dot(X)
  raw_loss = (scores - scores[y, np.arange(X.shape[1])]) + 1
  raw_loss[y, np.arange(X.shape[1])] = 0.0
  indiv_loss = np.maximum(raw_loss, 0.)
  data_loss = np.sum(indiv_loss)/X.shape[1] 
  reg_loss = 0.5 * reg * np.sum(W**2)
  loss = data_loss + reg_loss
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  d_raw_loss = np.ones(raw_loss.shape)/X.shape[1]
  d_raw_loss[raw_loss < 0] = 0.0
  d_scores = W * d_raw_loss
  d_scores = 1
  
  pass

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
