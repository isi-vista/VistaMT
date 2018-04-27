from collections import OrderedDict
import logging
import os

import numpy as np
import theano
# noinspection PyPep8Naming
import theano.tensor as T

from cnn.theano_utils import tparams_to_params, params_to_tparams

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def get_optimizer(name):
    name = name.lower()
    if name == 'sgd':
        return SGD()
    elif name == 'adam':
        return Adam()
    elif name == 'adadelta':
        return AdaDelta()
    elif name == 'nesterov':
        return NesterovMomentum()
    else:
        raise ValueError('unknown optimizer: ' + name)


class Optimizer:
    def get_updates(self, grads, tparams, learning_rate):
        raise NotImplementedError()

    def path_for_model(self, model_path):
        name = self.__class__.__name__.lower()
        return os.path.join(
            os.path.dirname(model_path), '{}-optimizer-{}'.format(
                name, os.path.basename(model_path)))

    def save(self, path):
        # only Adam saves state
        pass

    def load(self, path):
        # only Adam loads state
        pass


class SGD(Optimizer):
    DEFAULT_LEARNING_RATE = 0.1

    def get_updates(self, grads, tparams, learning_rate):
        updates = OrderedDict()
        for pname, grad in zip(tparams.keys(), grads):
            param = tparams[pname]
            updates[param] = param - learning_rate * grad
        return updates


class Adam(Optimizer):
    DEFAULT_LEARNING_RATE = 0.0002

    def __init__(self, b1=0.1, b2=0.001, e=1e-8):
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.opt_tparams = OrderedDict()

    def get_updates(self, grads, tparams, learning_rate):
        def init_param(name, default):
            value = self.opt_tparams.get(name, default)
            if name not in self.opt_tparams:
                self.opt_tparams[name] = value
            return value

        updates = []
        i = init_param('adam_i', theano.shared(np.float32(0.)))
        i_t = i + 1.
        fix1 = 1. - (1. - self.b1) ** i_t
        fix2 = 1. - (1. - self.b2) ** i_t
        lr_t = learning_rate * (T.sqrt(fix2) / fix1)
        for pname, g in zip(tparams.keys(), grads):
            p = tparams[pname]
            m = init_param('adam_m_' + pname, theano.shared(p.get_value() * 0.))
            v = init_param('adam_v_' + pname, theano.shared(p.get_value() * 0.))
            m_t = (self.b1 * g) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(g)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

    def save(self, path):
        log.info('Saving optimizer params to {}'.format(path))
        np.savez(path, **tparams_to_params(self.opt_tparams))

    def load(self, path):
        log.info('Loading optimizer params from {}'.format(path))
        self.opt_tparams = params_to_tparams(np.load(path))


class AdaDelta(Optimizer):
    DEFAULT_LEARNING_RATE = 1.0

    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon

    def get_updates(self, grads, tparams, learning_rate):
        updates = OrderedDict()
        one = T.constant(1)
        for pname, grad in zip(tparams.keys(), grads):
            param = tparams[pname]
            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)
            accu_new = self.rho * accu + (one - self.rho) * grad ** 2
            updates[accu] = accu_new
            update = (grad * T.sqrt(delta_accu + self.epsilon) /
                      T.sqrt(accu_new + self.epsilon))
            updates[param] = param - learning_rate * update
            delta_accu_new = self.rho * delta_accu + (one - self.rho) * update ** 2
            updates[delta_accu] = delta_accu_new
        return updates


class NesterovMomentum(Optimizer):
    DEFAULT_LEARNING_RATE = 0.25

    def __init__(self, momentum=0.3):
        self.momentum = momentum
        self.sgd = SGD()

    @staticmethod
    def _apply_nesterov_momentum(updates, momentum):
        params = updates.keys()
        updates = OrderedDict(updates)
        for param in params:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable)
            x = momentum * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = momentum * x + updates[param]
        return updates

    def get_updates(self, grads, tparams, learning_rate):
        updates = self.sgd.get_updates(grads, tparams, learning_rate)
        return self._apply_nesterov_momentum(updates, self.momentum)
