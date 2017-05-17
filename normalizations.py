import lasagne
from lasagne.layers import Layer
import theano.tensor as T
from lasagne.nonlinearities import rectify


# THIS CODE HAD TAKEN FROM:
# Instance normalization: https://github.com/jayanthkoushik/neural-style/blob/master/neural_style/fast_neural_style/transformer_net.py, @jayanthkoushik
# Weight normalization: https://github.com/openai/weightnorm/blob/master/lasagne/nn.py, @TimSalimans

class InstanceNormalization(Layer):
    def __init__(self, incoming, trainable=False, mean_only=False, **kwargs):
        super().__init__(incoming, **kwargs)
        self.scale = self.add_param(lasagne.init.Constant(1), (self.input_shape[1],), 'scale', trainable=trainable)
        self.shift = self.add_param(lasagne.init.Constant(0), (self.input_shape[1],), 'shift', trainable=trainable)
        self.mean_only = mean_only

    def get_output_for(self, input, **kwargs):
        input_mean = input.mean((2, 3), keepdims=True)
        input_inv_std = T.inv(T.sqrt(input.var((2, 3), keepdims=True) + 1e-4))
        if self.mean_only:
            normalized = (input - input_mean)
        else:
            normalized = (input - input_mean) * input_inv_std
        return self.scale.dimshuffle('x', 0, 'x', 'x') * normalized + self.shift.dimshuffle('x', 0, 'x', 'x')


class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), nonlinearity=rectify, transposed=False, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")
        if len(self.input_shape) == 4:
            self.axes_to_sum = (0, 2, 3)
            self.dimshuffle_args = ['x', 0, 'x', 'x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x', 0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim == 4:
            if transposed:
                W_axes_to_sum = (0, 2, 3)
                W_dimshuffle_args = ['x', 0, 'x', 'x']
            else:
                W_axes_to_sum = (1, 2, 3)
                W_dimshuffle_args = [0, 'x', 'x', 'x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x', 0]
        if g is not None:
            incoming.W = incoming.W_param * (
                self.g / T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum, keepdims=True))

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            stdv = T.sqrt(T.mean(T.square(input), axis=self.axes_to_sum))
            input /= stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m / stdv), (self.g, self.g / stdv)]
        elif hasattr(self, 'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(input)


def weight_norm(layer, transposed=False, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, transposed=transposed, **kwargs)
