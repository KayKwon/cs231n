import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
      score = X[i].dot(W)
      score -= np.max(score)    # stable softmax (prevent exponential to nan)
      correct_score = score[y[i]]
      denominator = np.sum(np.exp(score))
      softmax_p = np.exp(correct_score) / np.sum(np.exp(score))
      loss += -np.log(softmax_p)

      for c in range(num_class):
          if c == y[i]:
              dW[:, c] += X[i] * (softmax_p - 1) 
          else:
              dW[:, c] += X[i] * (np.exp(score[c]) / denominator)

  dW /= num_train
  dW += W * reg
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  score = X.dot(W)                               # 500 x 10
  score -= np.matrix(np.max(score, axis=1)).T    # stable softmax
  score = np.exp(score)
  y_score = score[np.arange(score.shape[0]), y]  # 500 x 1
  denom = np.sum(score, axis=1)                  # 500 x 1
  softmax_all = score / np.matrix(denom).T       # 500 x 10
  loss = np.sum(-np.log(y_score / denom))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  softmax_all[np.arange(softmax_all.shape[0]), y] -= 1
  dW = X.T.dot(softmax_all)
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

