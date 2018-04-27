from collections import OrderedDict

import theano


def params_to_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def tparams_to_params(tparams):
    params = OrderedDict()
    for kk, vv in tparams.items():
        params[kk] = vv.get_value()
    return params
