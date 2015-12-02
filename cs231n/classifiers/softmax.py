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
    probs = np.exp(scores)
    prob_sum = probs.sum()
    nprob_correct = probs[y[i]]/prob_sum
    loss += -np.log(nprob_correct)
    d_nprob_correct = -1./nprob_correct
    for cls in range(num_classes):
      if cls == y[i]:
        dprobs_cls = d_nprob_correct * (prob_sum - probs[y[i]])/prob_sum**2
      else:
        dprobs_cls = d_nprob_correct * -probs[y[i]]/prob_sum**2 
      dscores_cls = dprobs_cls * np.exp(scores[cls])
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
  probs = np.exp(scores)
  prob_sum = probs.sum(axis=0)
  nprob_correct = probs[y, col_range]/prob_sum
  loss = np.mean(-np.log(nprob_correct))
  loss += 0.5 * reg * np.sum(W*W)
  d_nprob_correct = -1./nprob_correct
  dprobs = np.tile(-probs[y, col_range], (num_classes, 1))
  dprobs[y, col_range] += prob_sum
  dprobs *= d_nprob_correct/prob_sum**2
  dscores = dprobs * probs
  dW = dscores.dot(X.T) 
  dW /= num_samples
  dW += reg * W

  return loss, dW
