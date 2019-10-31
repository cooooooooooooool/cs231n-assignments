from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - conv - batch normalization - relu - 2x2 max pool - conv - relu - conv - batch normalization - relu - 2x2 max pool - affine - dropout - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=64, filter_size=3,
                 dropout=1, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.use_dropout = dropout != 1
        self.reg = reg
        self.dtype = dtype

    # *************initionlizes parameters**********************
        
        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0.0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        #output_dim = (N, 64, H, W)
        self.params['W11'] = np.random.normal(0.0, weight_scale, (2 * num_filters, num_filters, filter_size, filter_size))
        self.params['b11'] = np.zeros(2 * num_filters)
        self.params['gamma1'] = np.ones(2 * num_filters)
        self.params['beta1'] = np.zeros(2 * num_filters)
        #output_dim = (N, 128, H//2, W//2)
        self.params['W2'] = np.random.normal(0.0,weight_scale,(4 * num_filters, 2 * num_filters, filter_size, filter_size))
        self.params['b2'] = np.zeros(4 * num_filters)
        #output_dim = (N, 256, H//2, W//2)
        self.params['W22'] = np.random.normal(0.0,weight_scale,(8 * num_filters, 4 * num_filters, filter_size, filter_size))
        self.params['b22'] = np.zeros(8 * num_filters)
        self.params['gamma2'] = np.ones(8 * num_filters)
        self.params['beta2'] = np.zeros(8 * num_filters)
        #output_dim = (N, 512, H//4, W//4)
        self.params['W3'] = np.random.normal(0.0, weight_scale, (8 * num_filters * H//4 * W//4, 4096))
        self.params['b3'] = np.zeros(4096)
        #output_dim = (N, 4096)
        self.params['W4'] = np.random.normal(0.0, weight_scale, (4096, num_classes))
        self.params['b4'] = np.zeros(num_classes)
        #output_dim = (N, 10)
        # *******************dropout***********************

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}


        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        self.bn_params=[]
        self.bn_params = [{'mode': 'train'} for i in range(2)]


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W11, b11 = self.params['W11'], self.params['b11']
        W2, b2 = self.params['W2'], self.params['b2']
        W22, b22 = self.params['W22'], self.params['b22']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        for bn_param in self.bn_params:
            bn_param['mode'] = mode

        # **************************FORWARD PASS********************************

        out, conv_cache1 = conv_relu_forward(X, W1, b1, conv_param)
        out, conbre_cache11 = conv_bn_relu_forward(out, W11, b11, self.params['gamma1'], self.params['beta1'], conv_param, self.bn_params[0])
        out, pool_cache1 = max_pool_forward_fast(out, pool_param)
        out, conv_cache2 = conv_relu_forward(out, W2, b2, conv_param)
        out, conbre_cache22 = conv_bn_relu_forward(out, W22, b22, self.params['gamma2'], self.params['beta2'], conv_param, self.bn_params[1])
        out, pool_cache2 = max_pool_forward_fast(out, pool_param)
        if self.use_dropout:
            out, affine_cache1 = affine_forward(out, W3, b3)
            out, drop_cache = dropout_forward(out, self.dropout_param)
        else:
            out, affine_cache1 = affine_forward(out, W3, b3) 
        scores, affine_cache2 = affine_forward(out, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}

         # **************************BACKWARD PASS********************************

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W11 * W11) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(W22 * W22)+ 0.5 * self.reg * np.sum(W3 * W3) + 0.5 * self.reg * np.sum(W4 * W4)
        dout, dW4, db4 = affine_backward(dscores,affine_cache2)
        if self.use_dropout:
            dout = dropout_backward(dout, drop_cache)
        dout, dW3, db3 = affine_backward(dout, affine_cache1)
        dout = max_pool_backward_fast(dout,pool_cache2)
        dout, dW22, db22, dgamma2, dbeta2 = conv_bn_relu_backward(dout, conbre_cache22)
        dout, dW2, db2 = conv_relu_backward(dout, conv_cache2)
        dout = max_pool_backward_fast(dout, pool_cache1)
        dout, dW11, db11, dgamma1, dbeta1 = conv_bn_relu_backward(dout, conbre_cache11)
        dout, dW1, db1 = conv_relu_backward(dout, conv_cache1)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1
        grads['W11'] = dW11 + self.reg * W11
        grads['b11'] = db11
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        grads['W22'] = dW22 + self.reg * W22
        grads['b22'] = db22
        grads['W3'] = dW3 + self.reg * W3
        grads['b3'] = db3
        grads['W4'] = dW4 + self.reg * W4
        grads['b4'] = db4
        grads['gamma1'] = dgamma1
        grads['beta1'] = dbeta1
        grads['gamma2'] = dgamma2
        grads['beta2'] = dbeta2

        return loss, grads


class SmallConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - batch normalization - relu - 2x2 max pool - conv - batch normalization - relu - 2x2 max pool - affine - dropout - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
                 dropout=1, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.use_dropout = dropout != 1
        self.reg = reg
        self.dtype = dtype

    # *************initionlizes parameters**********************
        
        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0.0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['gamma1'] = np.ones(num_filters)
        self.params['beta1'] = np.zeros(num_filters)
        #output_dim = (N, 64, H//2, W//2)
        self.params['W2'] = np.random.normal(0.0,weight_scale,(2*num_filters, num_filters, filter_size, filter_size))
        self.params['b2'] = np.zeros(2 * num_filters)
        self.params['gamma2'] = np.ones(2 * num_filters)
        self.params['beta2'] = np.zeros(2 * num_filters)
        #output_dim = (N, 128, H//4, W//4)
        self.params['W3'] = np.random.normal(0.0, weight_scale, (2 * num_filters * H//4 * W//4, 4096))
        self.params['b3'] = np.zeros(4096)
        #output_dim = (N, 4096)
        self.params['W4'] = np.random.normal(0.0, weight_scale, (4096, num_classes))
        self.params['b4'] = np.zeros(num_classes)
        #output_dim = (N, 10)
        # *******************dropout***********************

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}


        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        self.bn_params=[]
        self.bn_params = [{'mode': 'train'} for i in range(2)]


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # **************************FORWARD PASS********************************

        out, conbre_cache1 = conv_bn_relu_forward(X, W1, b1, self.params['gamma1'], self.params['beta1'], conv_param, self.bn_params[0])
        out, pool_cache1 = max_pool_forward_fast(out, pool_param)
        out, conbre_cache2 = conv_bn_relu_forward(out, W2, b2, self.params['gamma2'], self.params['beta2'], conv_param, self.bn_params[1])
        out, pool_cache2 = max_pool_forward_fast(out, pool_param)
        if self.use_dropout:
            out, affine_cache1 = affine_forward(out, W3, b3)
            out, drop_cache = dropout_forward(out, self.dropout_param)
        else:
            out, affine_cache1 = affine_forward(out, W3, b3) 
        scores, affine_cache2 = affine_forward(out, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}

         # **************************BACKWARD PASS********************************

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(W3 * W3) + 0.5 * self.reg * np.sum(W4 * W4)
        dout, dW4, db4 = affine_backward(dscores,affine_cache2)
        if self.use_dropout:
            dout = dropout_backward(dout, drop_cache)
        dout, dW3, db3 = affine_backward(dout, affine_cache1)
        dout = max_pool_backward_fast(dout,pool_cache2)
        dout, dW2, db2, dgamma2, dbeta2 = conv_bn_relu_backward(dout, conbre_cache2)
        dout = max_pool_backward_fast(dout, pool_cache1)
        dout, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_backward(dout, conbre_cache1)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        grads['W3'] = dW3 + self.reg * W3
        grads['b3'] = db3
        grads['W4'] = dW4 + self.reg * W4
        grads['b4'] = db4
        grads['gamma1'] = dgamma1
        grads['beta1'] = dbeta1
        grads['gamma2'] = dgamma2
        grads['beta2'] = dbeta2

        return loss, grads