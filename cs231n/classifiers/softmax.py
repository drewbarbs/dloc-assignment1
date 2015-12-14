import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_samples = X.shape[1]
  num_classes = W.shape[0]
  for i in range(num_samples):
    scores = W.dot(X[:, i])
    # numerical stability: subtracting same value from all scores
    # prior to exponentiating is equiv to multiplying top/bottom of softmax expression
    # by a constant
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    escore_sum = exp_scores.sum()
    nprob_correct = exp_scores[y[i]]/escore_sum
    loss += -np.log(nprob_correct)
    d_nprob_correct = -1./nprob_correct
    for cls in range(num_classes):
      if cls == y[i]:
        dexp_scores_cls = d_nprob_correct * (escore_sum - exp_scores[y[i]])/escore_sum**2
      else:
        dexp_scores_cls = d_nprob_correct * -exp_scores[y[i]]/escore_sum**2 
      dscores_cls = dexp_scores_cls * np.exp(scores[cls])
      dW[cls, :] += dscores_cls * X[:, i]
  loss /= num_samples
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_samples
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  num_classes, num_samples = W.shape[0], X.shape[1]
  col_range = np.arange(num_samples)
  scores = W.dot(X)
  scores -= np.max(scores, axis=0)
  exp_scores = np.exp(scores)
  escore_sum = exp_scores.sum(axis=0)
  loss = np.mean(-scores[y, col_range] + np.log(escore_sum))
  loss += 0.5 * reg * np.sum(W*W)
  dscores = exp_scores/escore_sum
  dscores[y, col_range] -= 1
  dW = dscores.dot(X.T) 
  dW /= num_samples
  dW += reg * W

  return loss, dW
