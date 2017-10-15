from collections import OrderedDict

import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, TransposedConv2DLayer, Conv2DLayer, \
    ReshapeLayer, ConcatLayer, FlattenLayer, NonlinearityLayer, ElemwiseMergeLayer, get_all_layers, dropout
from lasagne.layers import get_output, get_all_params
from lasagne.nonlinearities import rectify, sigmoid, tanh, LeakyRectify
from lasagne.objectives import squared_error
from lasagne.regularization import regularize_network_params
from lasagne.updates import adam

import utils
from normalizations import InstanceNormalization, weight_norm
from utils import one_hot

leaky_rectify = LeakyRectify(0.2)


def transform(x):
    f = False
    if len(x.shape) == 3:
        x, f = x[np.newaxis], True
    res = np.float32(x.copy())
    res = res.transpose((0, 3, 1, 2))
    res[:, :3] = res[:, :3] * 2 - 1
    if f: res = res[0]
    return res


def transform_inverse(x):
    f = False
    if len(x.shape) == 3:
        x, f = x[np.newaxis], True
    res = np.float32(x.copy())
    res[:, :3] = (res[:, :3] + 1) / 2
    res = res.transpose((0, 2, 3, 1))
    if f: res = res[0]
    return res


ANGLES1 = [20.0, 30.0]
ANGLES2 = [-175.0, -163.0, -151.0, -140.0, -128.0, -117.0, -105.0, -93.0, -82.0, -70.0, -59.0, -47.0, -35.0, -24.0,
           -12.0, 0.0, 11.0, 23.0, 34.0, 46.0, 58.0, 69.0, 81.0, 92.0, 104.0, 116.0, 127.0, 139.0, 150.0, 162.0, 174.0]
ANGLES = [(i, j) for i in ANGLES1 for j in ANGLES2]


class Model:
    def __init__(self, reg=False):
        self.reg = reg
        self.args = {}
        self.functions = {}
        self._build_gen()
        self._build_disc()
        self._build_fn()

    def gen(self, c, v, t):
        if not hasattr(c, 'shape') or len(c.shape) != 2:
            c = one_hot(c, 843)
        v = np.float32(v)
        t = np.float32(t)
        batch = {'c': c, 'v': v, 't': t}
        res = self.functions['gen'](*[batch[k] for k in self.args['gen']])
        return transform_inverse(res)

    def disc(self, x, c, v, t):
        c = one_hot(c, 843)
        v = np.float32(v)
        t = np.float32(t)
        x = transform(x)
        batch = {'c': c, 'v': v, 't': t, 'x': x}
        res = self.functions['disc'](*[batch[k] for k in self.args['disc']])
        return res

    def train_gen(self, c, v, t):
        c = one_hot(c, 843)
        v = np.float32(v)
        t = np.float32(t)
        batch = {'c': c, 'v': v, 't': t}
        res = np.array(self.functions['train_gen'](*[batch[k] for k in self.args['train_gen']]))
        assert not np.isnan(res.sum())
        return np.array(res)

    def train_disc(self, x, c, v, t):
        nc = c
        while (nc == c).any(): nc = np.random.randint(0, 843, len(c))
        nc = one_hot(nc, 843)
        c = one_hot(c, 843)

        v = np.float32(v)
        rands = []
        for i in range(len(v)): rands.append(np.random.randint(0, len(ANGLES)))
        nv = np.float32(utils.angle2sincos([ANGLES[i] for i in rands]))

        t = np.float32(t)
        nt = t.copy()
        np.random.shuffle(nt)

        x = transform(x)

        batch = {'x': x, 'c': c, 'v': v, 't': t, 'nc': nc, 'nt': nt, 'nv': nv}
        res = np.array(self.functions['train_disc'](*[batch[k] for k in self.args['train_disc']]))
        assert not np.isnan(res.sum())
        return np.array(res)

    def save(self, file):
        gen_params = lasagne.layers.get_all_param_values(self.gen_outputs.values())
        disc_params = lasagne.layers.get_all_param_values(self.disc_outputs.values())
        gen_update_params = [p.get_value() for p in self.gen_updates.keys()]
        disc_update_params = [p.get_value() for p in self.disc_updates.keys()]
        params = (gen_params, disc_params, gen_update_params, disc_update_params)
        utils.save(file, params)

    def load(self, file):
        params = utils.load(file)
        if len(params) == 4:
            gen_params, disc_params, gen_update_params, disc_update_params = utils.load(file)
            lasagne.layers.set_all_param_values(self.gen_outputs.values(), gen_params)
            lasagne.layers.set_all_param_values(self.disc_outputs.values(), disc_params)
            for p, value in zip(self.gen_updates.keys(), gen_update_params):
                p.set_value(value)
            for p, value in zip(self.disc_updates.keys(), disc_update_params):
                p.set_value(value)
        else:
            gen_params, disc_params = utils.load(file)
            lasagne.layers.set_all_param_values(self.gen_outputs.values(), gen_params)
            lasagne.layers.set_all_param_values(self.disc_outputs.values(), disc_params)

    def _build_gen(self):
        size = 64
        s, s2, s4, s8, s16 = size, size // 2, size // 4, size // 8, size // 16
        inputs = OrderedDict()
        inputs['c'] = InputLayer((None, 843))
        inputs['v'] = InputLayer((None, 4))
        inputs['t'] = InputLayer((None, 8))

        layer_c = inputs['c']
        layer_c = DenseLayer(layer_c, 512, nonlinearity=rectify)
        layer_c.params[layer_c.W].add('dense')
        layer_c = DenseLayer(layer_c, 512, nonlinearity=rectify)
        layer_c.params[layer_c.W].add('dense')

        layer_v = inputs['v']
        layer_v = DenseLayer(layer_v, 512, nonlinearity=rectify)
        layer_v.params[layer_v.W].add('dense')
        layer_v = DenseLayer(layer_v, 512, nonlinearity=rectify)
        layer_v.params[layer_v.W].add('dense')

        layer_t = inputs['t']
        layer_t = DenseLayer(layer_t, 512, nonlinearity=rectify)
        layer_t.params[layer_t.W].add('dense')
        layer_t = DenseLayer(layer_t, 512, nonlinearity=rectify)
        layer_t.params[layer_t.W].add('dense')

        layer = ConcatLayer([layer_c, layer_v, layer_t])
        layer = DenseLayer(layer, 1024, nonlinearity=rectify)
        layer.params[layer.W].add('dense')
        layer = DenseLayer(layer, 1024, nonlinearity=rectify)
        layer.params[layer.W].add('dense')

        layer = DenseLayer(layer, 768 * s16 * s16, nonlinearity=rectify)
        layer.params[layer.W].add('dense')
        layer = ReshapeLayer(layer, (-1, 768, s16, s16))

        layer = InstanceNormalization(layer, True)
        layer = weight_norm(
            TransposedConv2DLayer(layer, 384, 5, 2, 'same', output_size=(s8, s8), nonlinearity=None, b=None),
            transposed=True)
        if self.reg: layer = dropout(layer)
        layer = NonlinearityLayer(layer, rectify)
        layer = weight_norm(
            TransposedConv2DLayer(layer, 256, 5, 2, 'same', output_size=(s4, s4), nonlinearity=None, b=None),
            transposed=True)
        if self.reg: layer = dropout(layer)
        layer = NonlinearityLayer(layer, rectify)
        layer = weight_norm(
            TransposedConv2DLayer(layer, 192, 5, 2, 'same', output_size=(s2, s2), nonlinearity=None, b=None),
            transposed=True)
        if self.reg: layer = dropout(layer)
        layer = NonlinearityLayer(layer, rectify)

        layer_img = TransposedConv2DLayer(layer, 3, 5, 2, 'same', output_size=(s, s), nonlinearity=tanh)
        layer_msk = TransposedConv2DLayer(layer, 1, 5, 2, 'same', output_size=(s, s), nonlinearity=sigmoid)

        layer = ConcatLayer([layer_img, layer_msk])
        outputs = OrderedDict()
        outputs['x'] = layer
        self.gen_inputs = inputs
        self.gen_outputs = outputs

    def _build_disc(self):
        inputs = OrderedDict()
        inputs['x'] = InputLayer((None, 4, 64, 64))
        inputs['c'] = InputLayer((None, 843))
        inputs['v'] = InputLayer((None, 4))
        inputs['t'] = InputLayer((None, 8))

        layer_c = inputs['c']
        layer_c = DenseLayer(layer_c, 512, nonlinearity=leaky_rectify)
        layer_c.params[layer_c.W].add('dense')
        layer_c = (DenseLayer(layer_c, 512, nonlinearity=leaky_rectify))
        layer_c.params[layer_c.W].add('dense')

        layer_v = inputs['v']
        layer_v = DenseLayer(layer_v, 512, nonlinearity=leaky_rectify)
        layer_v.params[layer_v.W].add('dense')
        layer_v = (DenseLayer(layer_v, 512, nonlinearity=leaky_rectify))
        layer_v.params[layer_v.W].add('dense')

        layer_t = inputs['t']
        layer_t = DenseLayer(layer_t, 512, nonlinearity=leaky_rectify)
        layer_t.params[layer_t.W].add('dense')
        layer_t = (DenseLayer(layer_t, 512, nonlinearity=leaky_rectify))
        layer_t.params[layer_t.W].add('dense')

        layer_i = ConcatLayer([layer_c, layer_v, layer_t])
        layer_i = DenseLayer(layer_i, 1024, nonlinearity=leaky_rectify)
        layer_i.params[layer_i.W].add('dense')
        layer_i = DenseLayer(layer_i, 1024, nonlinearity=None)
        layer_i.params[layer_i.W].add('dense')

        layer_x = inputs['x']
        layer_x_n = layer_x
        layer_x = weight_norm(Conv2DLayer(layer_x_n, 64, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 64, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 128, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 256, 5, 2, 'same', nonlinearity=None, b=None))
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)

        layer_x = FlattenLayer(layer_x)
        layer_x = DenseLayer(layer_x, 1024, nonlinearity=leaky_rectify)
        layer_x.params[layer_x.W].add('dense')
        layer_x = DenseLayer(layer_x, 1024, nonlinearity=None)
        layer_x.params[layer_x.W].add('dense')

        layer = ElemwiseMergeLayer([layer_i, layer_x], T.mul)
        layer = ConcatLayer([layer, layer_x, layer_i])
        layer = DenseLayer(layer, 1024, nonlinearity=leaky_rectify)
        layer.params[layer.W].add('dense')

        layer_s = layer
        layer_s = DenseLayer(layer_s, 1, nonlinearity=None)
        layer_s.params[layer_s.W].add('dense')
        layer_s_0 = NonlinearityLayer(layer_s, nonlinearity=sigmoid)
        layer_s_1 = NonlinearityLayer(layer_s, nonlinearity=lambda x: x - T.log(1 + T.exp(x)))
        layer_s_2 = NonlinearityLayer(layer_s, nonlinearity=lambda x: -T.log(1 + T.exp(x)))

        outputs = OrderedDict()
        outputs['s'] = layer_s_0
        outputs['log(s)'] = layer_s_1
        outputs['log(1-s)'] = layer_s_2

        self.disc_inputs = inputs
        self.disc_outputs = outputs

        self.disc_inputs = inputs
        self.disc_outputs = outputs

    def _build_fn(self):
        c_var = self.gen_inputs['c'].input_var
        v_var = self.gen_inputs['v'].input_var
        t_var = self.gen_inputs['t'].input_var
        x_var = self.disc_inputs['x'].input_var
        neg_c_var = T.fmatrix()
        neg_t_var = T.fmatrix()
        neg_v_var = T.fmatrix()
        fx = get_output(self.gen_outputs['x'])

        s = get_output(self.disc_outputs['log(s)'], inputs={self.disc_inputs['x']: x_var,
                                                            self.disc_inputs['c']: c_var,
                                                            self.disc_inputs['v']: v_var,
                                                            self.disc_inputs['t']: t_var})

        fs, fs_1 = get_output([self.disc_outputs['log(1-s)'], self.disc_outputs['log(s)']],
                              inputs={self.disc_inputs['x']: fx,
                                      self.disc_inputs['c']: c_var,
                                      self.disc_inputs['v']: v_var,
                                      self.disc_inputs['t']: t_var})

        neg_s_c = get_output(self.disc_outputs['log(1-s)'], inputs={self.disc_inputs['x']: x_var,
                                                                    self.disc_inputs['c']: neg_c_var,
                                                                    self.disc_inputs['v']: v_var,
                                                                    self.disc_inputs['t']: t_var})

        neg_s_v = get_output(self.disc_outputs['log(1-s)'], inputs={self.disc_inputs['x']: x_var,
                                                                    self.disc_inputs['c']: c_var,
                                                                    self.disc_inputs['v']: neg_v_var,
                                                                    self.disc_inputs['t']: t_var})

        neg_s_t = get_output(self.disc_outputs['log(1-s)'], inputs={self.disc_inputs['x']: x_var,
                                                                    self.disc_inputs['c']: c_var,
                                                                    self.disc_inputs['v']: v_var,
                                                                    self.disc_inputs['t']: neg_t_var})

        disc_loss_s = -s.mean()
        disc_loss_fs = -fs.mean()
        neg_disc_loss_s = -neg_s_c.mean()
        neg_disc_loss_s += (-neg_s_t * (squared_error(neg_t_var, t_var))).mean()
        neg_disc_loss_s += (-neg_s_v * squared_error(neg_v_var, v_var)).mean()
        disc_loss_reg = regularize_network_params(self.disc_outputs['s'],
                                                  lambda x: T.mean(x ** 2),
                                                  tags={'regularizable': True, 'dense': True}).mean()
        disc_losses = [disc_loss_s, disc_loss_fs, neg_disc_loss_s]
        if self.reg: disc_losses.append(disc_loss_reg)

        gen_loss_s = -fs_1.mean()
        gen_loss_reg = regularize_network_params(self.gen_outputs['x'],
                                                 lambda x: T.mean(x ** 2),
                                                 tags={'regularizable': True, 'dense': True}).mean()
        gen_losses = [gen_loss_s]
        if self.reg: gen_losses.append(gen_loss_reg)
        self.lr = theano.shared(np.float32(2e-4))

        gen_params = get_all_params(self.gen_outputs.values(), trainable=True)
        self.gen_updates = adam(sum(gen_losses), gen_params, self.lr, beta1=0.5)

        disc_params = get_all_params(self.disc_outputs.values(), trainable=True)
        self.disc_updates = adam(sum(disc_losses), disc_params, self.lr, beta1=0.5)

        self.functions = {
            'gen': theano.function([self.gen_inputs['c'].input_var,
                                    self.gen_inputs['v'].input_var,
                                    self.gen_inputs['t'].input_var],
                                   get_output(self.gen_outputs['x'], deterministic=True)),
            'train_gen': theano.function([self.gen_inputs['c'].input_var,
                                          self.gen_inputs['v'].input_var,
                                          self.gen_inputs['t'].input_var],
                                         gen_losses,
                                         updates=self.gen_updates),

            'disc': theano.function([x_var, c_var, v_var, t_var],
                                    [get_output(self.disc_outputs['s'],
                                                inputs={self.disc_inputs['x']: x_var,
                                                        self.disc_inputs['c']: c_var,
                                                        self.disc_inputs['v']: v_var,
                                                        self.disc_inputs['t']: t_var}, deterministic=True),
                                     ]),

            'train_disc': theano.function([self.gen_inputs['c'].input_var,
                                           neg_c_var,
                                           neg_v_var,
                                           neg_t_var,
                                           self.gen_inputs['v'].input_var,
                                           self.gen_inputs['t'].input_var,
                                           x_var],
                                          disc_losses,
                                          updates=self.disc_updates),
        }

        self.args['gen'] = ('c', 'v', 't')
        self.args['train_gen'] = ('c', 'v', 't')
        self.args['disc'] = ('x', 'c', 'v', 't')
        self.args['train_disc'] = ('c', 'nc', 'nv', 'nt', 'v', 't', 'x')

    def build_classifier(self):
        layers = get_all_layers(self.disc_outputs['s'])
        prec_x_output = layers[-6]
        preclac_x = theano.function([layers[12].input_var], get_output(prec_x_output))
        prec_x_input = InputLayer((None, 1024))
        ewm = layers[-5]
        ewm.input_layers[1] = prec_x_input
        cnc = layers[-4]
        cnc.input_layers[1] = prec_x_input
        prove_output = self.disc_outputs['s']
        prove = theano.function([layers[0].input_var, layers[3].input_var, layers[6].input_var, prec_x_input.input_var],
                                get_output(prove_output))
        ewm.input_layers[1] = prec_x_output
        cnc.input_layers[1] = prec_x_output

        def classify(x, cs, vs, ts):
            assert len(cs) == len(vs) == len(ts)
            n = len(cs)
            if (classify.last_x == x).all():
                f = classify.last_f
            else:
                f = preclac_x(transform(x)[np.newaxis])
            classify.last_x = x
            classify.last_f = f
            f = f.repeat(n, 0)
            result = prove(one_hot(cs, 843), np.float32(vs), np.float32(ts), f)
            return result.flatten()

        classify.last_x = np.inf
        classify.last_f = np.inf
        self.classify = classify


class ModelSplit(Model):
    def _build_disc(self):
        inputs = OrderedDict()
        inputs['x'] = InputLayer((None, 4, 64, 64))
        inputs['c'] = InputLayer((None, 843))
        inputs['v'] = InputLayer((None, 4))
        inputs['t'] = InputLayer((None, 8))

        layer_c = inputs['c']
        layer_c = DenseLayer(layer_c, 512, nonlinearity=leaky_rectify)
        layer_c.params[layer_c.W].add('dense')
        layer_c = (DenseLayer(layer_c, 512, nonlinearity=leaky_rectify))
        layer_c.params[layer_c.W].add('dense')

        layer_v = inputs['v']
        layer_v = DenseLayer(layer_v, 512, nonlinearity=leaky_rectify)
        layer_v.params[layer_v.W].add('dense')
        layer_v = (DenseLayer(layer_v, 512, nonlinearity=leaky_rectify))
        layer_v.params[layer_v.W].add('dense')

        layer_t = inputs['t']
        layer_t = DenseLayer(layer_t, 512, nonlinearity=leaky_rectify)
        layer_t.params[layer_t.W].add('dense')
        layer_t = (DenseLayer(layer_t, 512, nonlinearity=leaky_rectify))
        layer_t.params[layer_t.W].add('dense')

        layer_i = ConcatLayer([layer_c, layer_v, layer_t])
        layer_i = DenseLayer(layer_i, 1024, nonlinearity=leaky_rectify)
        layer_i.params[layer_i.W].add('dense')
        layer_i = DenseLayer(layer_i, 1024, nonlinearity=None)
        layer_i.params[layer_i.W].add('dense')
        layer_i = NonlinearityLayer(layer_i, lambda x: x/T.sqrt(T.sum(T.square(x), axis=1, keepdims=True)))

        layer_x = inputs['x']
        layer_x_n = layer_x
        layer_x = weight_norm(Conv2DLayer(layer_x_n, 64, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 64, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 128, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 256, 5, 2, 'same', nonlinearity=None, b=None))
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)

        layer_x = FlattenLayer(layer_x)
        layer_x = DenseLayer(layer_x, 1024, nonlinearity=leaky_rectify)
        layer_x.params[layer_x.W].add('dense')

        layer_x = DenseLayer(layer_x, 1024, nonlinearity=None)
        layer_x.params[layer_x.W].add('dense')
        layer_x = NonlinearityLayer(layer_x, lambda x: x/T.sqrt(T.sum(T.square(x), axis=1, keepdims=True)))

        layer = ElemwiseMergeLayer([layer_i, layer_x], T.mul)
        layer = ConcatLayer([layer, layer_x, layer_i])
        layer = DenseLayer(layer, 1024, nonlinearity=leaky_rectify)
        layer.params[layer.W].add('dense')

        layer_r = DenseLayer(layer, 1024, nonlinearity=leaky_rectify)
        layer_r.params[layer_r.W].add('dense')
        layer_r = DenseLayer(layer_r, 1, nonlinearity=None)
        layer_r.params[layer_r.W].add('dense')
        layer_r_0 = NonlinearityLayer(layer_r, nonlinearity=sigmoid)
        layer_r_1 = NonlinearityLayer(layer_r, nonlinearity=lambda x: x - T.log(1 + T.exp(x)))
        layer_r_2 = NonlinearityLayer(layer_r, nonlinearity=lambda x: -T.log(1 + T.exp(x)))

        layer_s = DenseLayer(layer, 1024, nonlinearity=leaky_rectify)
        layer_s.params[layer_s.W].add('dense')
        layer_s = DenseLayer(layer_s, 1, nonlinearity=None)
        layer_s.params[layer_s.W].add('dense')
        layer_s_0 = NonlinearityLayer(layer_s, nonlinearity=sigmoid)
        layer_s_1 = NonlinearityLayer(layer_s, nonlinearity=lambda x: x - T.log(1 + T.exp(x)))
        layer_s_2 = NonlinearityLayer(layer_s, nonlinearity=lambda x: -T.log(1 + T.exp(x)))

        outputs = OrderedDict()
        outputs['s'] = layer_s_0
        outputs['log(s)'] = layer_s_1
        outputs['log(1-s)'] = layer_s_2
        outputs['r'] = layer_r_0
        outputs['log(r)'] = layer_r_1
        outputs['log(1-r)'] = layer_r_2

        self.disc_inputs = inputs
        self.disc_outputs = outputs

    def _build_fn(self):
        c_var = self.gen_inputs['c'].input_var
        v_var = self.gen_inputs['v'].input_var
        t_var = self.gen_inputs['t'].input_var
        x_var = self.disc_inputs['x'].input_var
        neg_c_var = T.fmatrix()
        neg_t_var = T.fmatrix()
        neg_v_var = T.fmatrix()
        fx = get_output(self.gen_outputs['x'], deterministic=False)

        s, s_inv, r, r_inv = get_output([self.disc_outputs['log(s)'], self.disc_outputs['log(1-s)'],
                                         self.disc_outputs['log(r)'], self.disc_outputs['log(1-r)']],
                                        inputs={self.disc_inputs['x']: x_var,
                                                self.disc_inputs['c']: c_var,
                                                self.disc_inputs['v']: v_var,
                                                self.disc_inputs['t']: t_var}, deterministic=False)

        gs, gs_inv, gr, gr_inv = get_output([self.disc_outputs['log(s)'], self.disc_outputs['log(1-s)'],
                                             self.disc_outputs['log(r)'], self.disc_outputs['log(1-r)']],
                                            inputs={self.disc_inputs['x']: fx,
                                                    self.disc_inputs['c']: c_var,
                                                    self.disc_inputs['v']: v_var,
                                                    self.disc_inputs['t']: t_var}, deterministic=False)

        # here mb better to merge
        neg_s_c = get_output(self.disc_outputs['log(1-s)'], inputs={self.disc_inputs['x']: x_var,
                                                                    self.disc_inputs['c']: neg_c_var,
                                                                    self.disc_inputs['v']: v_var,
                                                                    self.disc_inputs['t']: t_var}, deterministic=False)

        neg_s_v = get_output(self.disc_outputs['log(1-s)'], inputs={self.disc_inputs['x']: x_var,
                                                                    self.disc_inputs['c']: c_var,
                                                                    self.disc_inputs['v']: neg_v_var,
                                                                    self.disc_inputs['t']: t_var}, deterministic=False)

        neg_s_t = get_output(self.disc_outputs['log(1-s)'], inputs={self.disc_inputs['x']: x_var,
                                                                    self.disc_inputs['c']: c_var,
                                                                    self.disc_inputs['v']: v_var,
                                                                    self.disc_inputs['t']: neg_t_var}, deterministic=False)

        disc_loss_s, disc_loss_r = -s.mean(), -r.mean()
        disc_loss_gr = -gr_inv.mean()
        disc_loss_neg = -neg_s_c.mean()
        disc_loss_neg += (-neg_s_t * (squared_error(neg_t_var, t_var))).mean()
        disc_loss_neg += (-neg_s_v * squared_error(neg_v_var, v_var)).mean()
        disc_loss_reg = regularize_network_params(self.disc_outputs['s'],
                                                  lambda x: T.mean(x ** 2),
                                                  tags={'regularizable': True, 'dense': True}).mean()
        disc_losses = [disc_loss_s, disc_loss_neg, disc_loss_r, disc_loss_gr]
        if self.reg: disc_losses.append(disc_loss_reg)

        gen_loss_gs, gen_loss_gr = -gs.mean(), -gr.mean()
        gen_loss_reg = regularize_network_params(self.gen_outputs['x'],
                                                 lambda x: T.mean(x ** 2),
                                                 tags={'regularizable': True, 'dense': True}).mean()
        gen_losses = [gen_loss_gs, gen_loss_gr]
        if self.reg: gen_losses.append(gen_loss_reg)
        self.lr = theano.shared(np.float32(2e-4))

        gen_params = get_all_params(self.gen_outputs.values(), trainable=True)
        self.gen_updates = adam(sum(gen_losses), gen_params, self.lr, beta1=0.5)

        disc_params = get_all_params(self.disc_outputs.values(), trainable=True)
        self.disc_updates = adam(sum(disc_losses), disc_params, self.lr, beta1=0.5)

        self.functions = {
            'gen': theano.function([self.gen_inputs['c'].input_var,
                                    self.gen_inputs['v'].input_var,
                                    self.gen_inputs['t'].input_var],
                                   get_output(self.gen_outputs['x'], deterministic=True)),
            'train_gen': theano.function([self.gen_inputs['c'].input_var,
                                          self.gen_inputs['v'].input_var,
                                          self.gen_inputs['t'].input_var],
                                         gen_losses,
                                         updates=self.gen_updates),

            'disc': theano.function([x_var, c_var, v_var, t_var],
                                    [get_output(self.disc_outputs['s'],
                                                inputs={self.disc_inputs['x']: x_var,
                                                        self.disc_inputs['c']: c_var,
                                                        self.disc_inputs['v']: v_var,
                                                        self.disc_inputs['t']: t_var}, deterministic=True),
                                     ]),

            'train_disc': theano.function([self.gen_inputs['c'].input_var,
                                           neg_c_var,
                                           neg_v_var,
                                           neg_t_var,
                                           self.gen_inputs['v'].input_var,
                                           self.gen_inputs['t'].input_var,
                                           x_var],
                                          disc_losses,
                                          updates=self.disc_updates),
        }

        self.args['gen'] = ('c', 'v', 't')
        self.args['train_gen'] = ('c', 'v', 't')
        self.args['disc'] = ('x', 'c', 'v', 't')
        self.args['train_disc'] = ('c', 'nc', 'nv', 'nt', 'v', 't', 'x')


class ModelWOEMM(ModelSplit):
    def _build_disc(self):
        inputs = OrderedDict()
        inputs['x'] = InputLayer((None, 4, 64, 64))
        inputs['c'] = InputLayer((None, 843))
        inputs['v'] = InputLayer((None, 4))
        inputs['t'] = InputLayer((None, 8))

        layer_c = inputs['c']
        layer_c = DenseLayer(layer_c, 512, nonlinearity=leaky_rectify)
        layer_c.params[layer_c.W].add('dense')
        layer_c = (DenseLayer(layer_c, 512, nonlinearity=leaky_rectify))
        layer_c.params[layer_c.W].add('dense')

        layer_v = inputs['v']
        layer_v = DenseLayer(layer_v, 512, nonlinearity=leaky_rectify)
        layer_v.params[layer_v.W].add('dense')
        layer_v = (DenseLayer(layer_v, 512, nonlinearity=leaky_rectify))
        layer_v.params[layer_v.W].add('dense')

        layer_t = inputs['t']
        layer_t = DenseLayer(layer_t, 512, nonlinearity=leaky_rectify)
        layer_t.params[layer_t.W].add('dense')
        layer_t = (DenseLayer(layer_t, 512, nonlinearity=leaky_rectify))
        layer_t.params[layer_t.W].add('dense')

        layer_i = ConcatLayer([layer_c, layer_v, layer_t])
        layer_i = DenseLayer(layer_i, 1024, nonlinearity=leaky_rectify)
        layer_i.params[layer_i.W].add('dense')
        layer_i = DenseLayer(layer_i, 1024, nonlinearity=None)
        layer_i.params[layer_i.W].add('dense')

        layer_x = inputs['x']
        layer_x_n = layer_x
        layer_x = weight_norm(Conv2DLayer(layer_x_n, 64, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 64, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 128, 5, 2, 'same', nonlinearity=None, b=None))
        if self.reg: layer_x = dropout(layer_x)
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)
        layer_x = weight_norm(Conv2DLayer(layer_x, 256, 5, 2, 'same', nonlinearity=None, b=None))
        layer_x = NonlinearityLayer(layer_x, leaky_rectify)

        layer_x = FlattenLayer(layer_x)
        layer_x = DenseLayer(layer_x, 1024, nonlinearity=leaky_rectify)
        layer_x.params[layer_x.W].add('dense')

        layer_x = DenseLayer(layer_x, 1024, nonlinearity=None)
        layer_x.params[layer_x.W].add('dense')

        layer = ConcatLayer([layer_x, layer_i])
        layer = DenseLayer(layer, 1024, nonlinearity=leaky_rectify)
        layer.params[layer.W].add('dense')

        layer_r = DenseLayer(layer, 1024, nonlinearity=leaky_rectify)
        layer_r.params[layer_r.W].add('dense')
        layer_r = DenseLayer(layer_r, 1, nonlinearity=None)
        layer_r.params[layer_r.W].add('dense')
        layer_r_0 = NonlinearityLayer(layer_r, nonlinearity=sigmoid)
        layer_r_1 = NonlinearityLayer(layer_r, nonlinearity=lambda x: x - T.log(1 + T.exp(x)))
        layer_r_2 = NonlinearityLayer(layer_r, nonlinearity=lambda x: -T.log(1 + T.exp(x)))

        layer_s = DenseLayer(layer, 1024, nonlinearity=leaky_rectify)
        layer_s.params[layer_s.W].add('dense')
        layer_s = DenseLayer(layer_s, 1, nonlinearity=None)
        layer_s.params[layer_s.W].add('dense')
        layer_s_0 = NonlinearityLayer(layer_s, nonlinearity=sigmoid)
        layer_s_1 = NonlinearityLayer(layer_s, nonlinearity=lambda x: x - T.log(1 + T.exp(x)))
        layer_s_2 = NonlinearityLayer(layer_s, nonlinearity=lambda x: -T.log(1 + T.exp(x)))

        outputs = OrderedDict()
        outputs['s'] = layer_s_0
        outputs['log(s)'] = layer_s_1
        outputs['log(1-s)'] = layer_s_2
        outputs['r'] = layer_r_0
        outputs['log(r)'] = layer_r_1
        outputs['log(1-r)'] = layer_r_2

        self.disc_inputs = inputs
        self.disc_outputs = outputs
