import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.layers.core import Layer
from keras.layers import Reshape

from logging import warn

def Input(shape):
    """Return no-op layer that can be used as an input layer."""
    return Reshape(shape, input_shape=shape)

class FixedEmbedding(Layer):
    """Embedding with fixed weights.

    Modified from keras/layers/embeddings.py in Keras (http://keras.io).

    WARNING: this is experimental and not fully tested, use at your
    own risk.
    """
    input_ndim = 2

    def __init__(self, input_dim, output_dim, weights, input_length=None,
                 mask_zero=False, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.mask_zero = mask_zero
        self.dropout = dropout

        if (not isinstance(weights, list) or len(weights) != 1 or
            weights[0].shape != (input_dim, output_dim)):
            raise ValueError('weights must be a list with single element'
                             ' with shape (input_dim, output_dim).')
        self.initial_weights = weights

        lrmul = ('W_learning_rate_multiplier', 'b_learning_rate_multiplier')
        if any(a in kwargs for a in lrmul):
            for a in lrmul:
                kwargs.pop(a, None)
            warn('FixedEmbedding: learning rate multiplier ignored')

        kwargs['input_shape'] = (self.input_dim,)
        super(FixedEmbedding, self).__init__(**kwargs)

    def build(self):
        self.input = K.placeholder(shape=(self.input_shape[0],
                                          self.input_length),
                                   dtype='int32')
        self.W = K.variable(self.initial_weights[0])
        self.trainable_weights = []
        self.regularizers = []

    def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(X, 0)

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_length, self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dropout:
            raise NotImplementedError()     # TODO
        out = K.gather(self.W, X)
        return out

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "output_dim": self.output_dim,
                  "input_length": self.input_length,
                  "mask_zero": self.mask_zero,
                  "dropout": self.dropout}
        base_config = super(FixedEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class UnbiasedDense(Layer):
    '''Regular fully connected NN layer without bias term.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list containing numpy array to set as initial weights.
            The list should have a single element of shape
            `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        W_learning_rate_multiplier: Multiplier (between 0.0 and 1.0) applied to the 
            learning rate of the main weights matrix.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 W_learning_rate_multiplier=None,
                 input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(UnbiasedDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))

        self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(K.dot(X, self.W))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'input_dim': self.input_dim}
        base_config = super(UnbiasedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
