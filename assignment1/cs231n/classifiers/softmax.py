from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_trian = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_trian):
      scores=X[i].dot(W)
      scores -= np.max(scores)
      den = np.sum(np.exp(scores))
      p=lambda k : np.exp(scores[k])/den
      loss += -np.log(p(y[i]))
      for j in range(num_class):
        if j==y[i]:
          dW[:,j] += (p(j)-1)*X[i,:].T
        else:
          dW[:,j] += p(j)*X[i,:].T
    #print('n_loss is %f'%loss)//for checking data
    loss /= num_trian
    loss += 0.5*reg*np.sum(W*W)
    dW= dW/num_trian+reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_trian = X.shape[0] # N
    scores = X.dot(W) #N x C
    scores_max = np.max(scores,axis = 1,keepdims= True)
    scores -= scores_max
    exp_scores = np.exp(scores)
    den = np.sum(exp_scores,axis=1)
    loss= np.sum(-np.log(exp_scores[np.arange(num_trian),y]/den))
    #print('v_loss is %f'%loss) //for checking data
    loss= loss/num_trian + 0.5*reg*np.sum(W*W)
    p = exp_scores/np.reshape(den, (num_trian, 1))
    p[np.arange(num_trian),y]=p[np.arange(num_trian),y]-1
    dW = X.T.dot(p)
    dW = dW/num_trian+reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
