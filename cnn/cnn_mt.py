# convolutional neural machine translation

import copy
import logging
from collections import OrderedDict
from math import sqrt

import numpy as np
import theano
# noinspection PyPep8Naming
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cnn.theano_utils import params_to_tparams, tparams_to_params
from cnn.vocab import Vocab
from cnn.conv1d import conv1d_mc0

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

tconfig = theano.config
np.random.seed(1234)
trng = RandomStreams(1234)


# noinspection PyPep8Naming
class CNMT(object):
    def __init__(self, x_vocab, y_vocab, config):
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        log.info('Initializing model...')
        self.config = config
        params = self._init_params()
        self.tparams = params_to_tparams(params)
        self.num_decoder_stacks = len(self.config.decoder_arch)
        num_decoder_layers = 0
        for a in self.config.decoder_arch:
            num_decoder_layers += a[0]
        self.num_decoder_layers = num_decoder_layers
        log.info('ModelConfiguration: %s', self.config.to_json())

    def load_params(self, filename):
        log.info('Loading params from {}'.format(filename))
        params = {k: v for k, v in np.load(filename).items()
                  if k.startswith('enc_') or k.startswith('dec_')}
        self.tparams = params_to_tparams(params)
        log.info('END Loading params')

    def save_params(self, filename):
        log.info('Saving params to {}'.format(filename))
        np.savez(filename, **tparams_to_params(self.tparams))
        log.info('END Saving params')

    def build_trainer(self, optimizer):
        tparams = self.tparams
        x, x_mask, y_lshifted, y_rshifted, y_mask, cost, train, lr = \
            self._build_model(tparams)
        grads = []
        for kk, vv in tparams.items():
            grads.append(T.grad(cost, vv))
        grads = self._clip_gradients(grads, 1.)
        updates = optimizer.get_updates(grads, tparams, learning_rate=lr)
        f_train = theano.function(
            inputs=[x, x_mask, y_lshifted, y_rshifted, y_mask, train, lr],
            outputs=cost, updates=updates)
        f_compute_cost = theano.function(
            inputs=[x, x_mask, y_lshifted, y_rshifted, y_mask, train],
            outputs=cost)
        f_init, f_next = self._build_sampler(tparams)

        def train(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_, lr_):
            return f_train(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_, 1, lr_)

        def compute_cost(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_):
            return f_compute_cost(x_, x_mask_, y_lshifted_, y_rshifted_, y_mask_, 0)

        def predict(x_, max_words_):
            return self._gen_sample(f_init, f_next, x_, max_words_)

        return train, compute_cost, predict

    def build_predictor(self):
        tparams = self.tparams
        f_init, f_next = self._build_sampler(tparams)

        def predict(x, beam_width, max_words):
            samples, costs = self._gen_nbest(f_init, f_next, x, beam_width,
                                             max_words)
            best_sample = None
            best_cost = np.finfo(np.float32).max
            for c, s in zip(costs, samples):
                cost = c / len(s)
                if cost < best_cost:
                    best_cost = cost
                    best_sample = s
            return best_sample, best_cost

        return predict

    def build_batch_predictor(self):
        tparams = self.tparams
        f_sample = self._build_batch_sampler(tparams)

        def _trim_sentence(s):
            result = []
            for w in s:
                result.append(w)
                if w == self.y_vocab.lookup(Vocab.SENT_END):
                    break
            return result

        def predict(x, x_mask, n):
            decoded = f_sample(x, x_mask, n)
            result = []
            for i in range(decoded.shape[0]):
                result.append(_trim_sentence(decoded[i]))
            return result

        return predict

    def _init_params(self):
        params = OrderedDict()
        self._init_encoder(params, 'enc', self.config.dropout_rate)
        self._init_decoder(params, 'dec', self.config.dropout_rate)
        return params

    @staticmethod
    def _prepend(prefix, name):
        return '{}_{}'.format(prefix, name)

    @staticmethod
    def _zero_weights(size):
        w = np.zeros(size).astype(tconfig.floatX)
        return w

    @staticmethod
    def _normal_weights(size, scale=0.01):
        w = np.random.normal(size=size, scale=scale).astype(tconfig.floatX)
        return w

    @staticmethod
    def _linear(x):
        return x

    def _dropout_layer(self, layer_in, train, dropout_shape=None):
        dropout_rate = self.config.dropout_rate
        if dropout_shape is None:
            dropout_shape = layer_in.shape
        retain_rate = 1 - dropout_rate
        result = T.switch(
            train,
            layer_in * trng.binomial(dropout_shape, p=retain_rate, n=1,
                                     dtype=layer_in.dtype),
            layer_in * retain_rate)
        return result

    def _emb_layer(self, tparams, layer_in, prefix):
        return tparams[self._prepend(prefix, 'E')][layer_in]

    def _init_emb_layer(self, params, prefix, voc_size):
        params[self._prepend(prefix, 'E')] = self._normal_weights(
            size=(voc_size, self.config.emb_dim), scale=0.1)

    def _ff_layer(self, tparams, layer_in, prefix, activation):
        return activation(T.dot(
            layer_in, tparams[self._prepend(prefix, 'W')])
                          + tparams[self._prepend(prefix, 'b')])

    def _init_ff_layer(self, params, prefix, d_in, d_out, dropout=0):
        params[self._prepend(prefix, 'W')] = self._normal_weights(
            size=(d_in, d_out), scale=sqrt((1.0 - dropout) / d_in))
        params[self._prepend(prefix, 'b')] = self._zero_weights((d_out,))

    @staticmethod
    def _convolution_block(W, b, Wg, bg, block_in):
        block_in = block_in.dimshuffle(0, 2, 1)
        block_out = conv1d_mc0(block_in, W)
        block_out = block_out.dimshuffle(0, 2, 1)
        block_out = block_out + b
        gate_out = conv1d_mc0(block_in, Wg)
        gate_out = gate_out.dimshuffle(0, 2, 1)
        gate_out = gate_out + bg
        return block_out * T.nnet.sigmoid(gate_out)

    def _encoder(self, tparams, x, prefix, train):
        arch = self.config.encoder_arch
        # embed source
        x_emb = self._emb_layer(tparams, x, self._prepend(prefix, 'w_emb'))
        # dropout
        x_emb = self._dropout_layer(x_emb, train)
        # transform input
        conv_in = self._ff_layer(tparams, x_emb, self._prepend(prefix, 'prj_in'),
                                 self._linear)
        # convolutions
        conv_out = None
        in_dim = arch[0][2]
        for idx, spec in enumerate(arch):
            stack_name = 'stack_' + str(idx)
            depth, width, dim = spec
            if dim != in_dim:
                # transform input for this stack
                prj_name = 'stack_prj_in_' + str(idx)
                conv_in = self._ff_layer(tparams, conv_in,
                                         self._prepend(prefix, prj_name), self._linear)
            # convolution stack
            conv_out = self._encoder_stack(tparams, conv_in, width, dim,
                                           self._prepend(prefix, stack_name), train)
            in_dim = dim
            conv_in = conv_out
        # transform output
        ctx = self._ff_layer(tparams, conv_out,
                             self._prepend(prefix, 'prj_out'), self._linear)
        # scale gradients
        ctx = theano.gradient.grad_scale(
            ctx, 1. / (2 * self.num_decoder_layers))
        # context plus word embedding
        ctx_plus_emb = (ctx + x_emb) / sqrt(2)
        return ctx, ctx_plus_emb

    def _encoder_stack(self, tparams, stack_in, width, dim, prefix, train):
        input_shape = stack_in.shape
        enc_out, _ = theano.scan(self._encoder_block,
                                 sequences=(tparams[self._prepend(prefix, 'W')],
                                            tparams[self._prepend(prefix, 'b')],
                                            tparams[self._prepend(prefix, 'Wg')],
                                            tparams[self._prepend(prefix, 'bg')]),
                                 outputs_info=stack_in,
                                 non_sequences=(width, dim, input_shape, train))
        return enc_out[-1]

    def _encoder_block(self, W, b, Wg, bg, block_in, width, dim, input_shape,
                       train):
        batch_size = input_shape[0]
        # input dropout
        conv_in = self._dropout_layer(block_in, train,
                                      dropout_shape=input_shape)
        # pad borders
        padding = T.zeros((batch_size, width // 2, dim))
        conv_in = T.concatenate((padding, conv_in), axis=1)
        conv_in = T.concatenate((conv_in, padding), axis=1)
        # convolution_block
        conv_out = self._convolution_block(W, b, Wg, bg, conv_in)
        # residual connections
        block_out = (conv_out + block_in) / sqrt(2)
        return block_out

    def _init_encoder_stack(self, params, prefix, width, depth, dim, dropout):
        params[self._prepend(prefix, 'W')] = self._normal_weights(
            size=(depth, dim, dim, width),
            scale=sqrt((4.0 * (1.0 - dropout)) / (dim * width)))
        params[self._prepend(prefix, 'b')] = self._zero_weights(
            (depth, dim))
        params[self._prepend(prefix, 'Wg')] = self._normal_weights(
            size=(depth, dim, dim, width),
            scale=sqrt((4.0 * (1.0 - dropout)) / (dim * width)))
        params[self._prepend(prefix, 'bg')] = self._zero_weights(
            (depth, dim))

    def _init_encoder(self, params, prefix, dropout):
        arch = self.config.encoder_arch
        self._init_emb_layer(params, self._prepend(prefix, 'w_emb'),
                             self.x_vocab.size())
        in_dim = arch[0][2]
        self._init_ff_layer(params, self._prepend(prefix, 'prj_in'),
                            self.config.emb_dim, in_dim,
                            dropout=dropout)
        dim = None
        for idx, spec in enumerate(arch):
            stack_name = 'stack_' + str(idx)
            depth, width, dim = spec
            if dim != in_dim:
                prj_name = 'stack_prj_in_' + str(idx)
                self._init_ff_layer(params, self._prepend(prefix, prj_name),
                                    in_dim, dim)
            self._init_encoder_stack(params, self._prepend(prefix, stack_name),
                                     width, depth, dim, dropout)
            in_dim = dim
        self._init_ff_layer(params, self._prepend(prefix, 'prj_out'),
                            dim, self.config.emb_dim)

    @staticmethod
    def _attention_block(W_ATT_in, b_ATT_in, W_ATT_out, b_ATT_out,
                         ctx, ctx_plus_emb, x_mask, prev_w_emb, state_pre_attn):
        state_attn = T.dot(state_pre_attn, W_ATT_in) + b_ATT_in
        state_attn = (state_attn + prev_w_emb) / sqrt(2)
        x_len = x_mask.sum(axis=1).dimshuffle(0, 'x', 'x')
        alpha = T.batched_dot(state_attn, ctx.dimshuffle(0, 2, 1))
        alpha = T.switch(x_mask.dimshuffle(0, 'x', 1), alpha, np.NINF)
        alpha = alpha - alpha.max(2, keepdims=True)
        alpha = T.exp(alpha)
        alpha = alpha / alpha.sum(2, keepdims=True)
        ctx_contrib = alpha.dimshuffle(0, 1, 2, 'x') * ctx_plus_emb.dimshuffle(0, 'x', 1, 2)
        ctx_contrib = ctx_contrib.sum(axis=2)
        ctx_contrib = ctx_contrib * x_len / T.sqrt(x_len)
        state_post_attn = T.dot(ctx_contrib, W_ATT_out) + b_ATT_out
        return state_post_attn

    def _decoder(self, tparams, y_in, ctx, ctx_plus_emb, x_mask, prefix, train):
        arch = self.config.decoder_arch
        # embed target
        y_emb = self._emb_layer(tparams, y_in, self._prepend(prefix, 'w_emb'))
        # dropout
        y_emb = self._dropout_layer(y_emb, train)
        # transform input
        conv_in = self._ff_layer(tparams, y_emb, self._prepend(prefix, 'prj_in'),
                                 self._linear)
        # convolutions
        conv_out = None
        in_dim = arch[0][2]
        for idx, spec in enumerate(arch):
            stack_name = 'stack_' + str(idx)
            depth, width, dim = spec
            if dim != in_dim:
                # transform input for this stack
                prj_name = 'stack_prj_in_' + str(idx)
                conv_in = self._ff_layer(tparams, conv_in,
                                         self._prepend(prefix, prj_name), self._linear)
            # convolution stack
            conv_out = self._decoder_stack(tparams, conv_in, ctx, ctx_plus_emb,
                                           x_mask, y_emb, width, dim, self._prepend(prefix, stack_name),
                                           train)
            in_dim = dim
            conv_in = conv_out
        # transform output
        dec_out = self._ff_layer(tparams, conv_out, self._prepend(prefix,
                                                                  'prj_out_1'), self._linear)
        # dropout
        dec_out = self._dropout_layer(dec_out, train)
        # output layer
        logit = self._ff_layer(tparams, dec_out, self._prepend(prefix,
                                                               'prj_out_2'), self._linear)
        return logit

    def _decoder_stack(self, tparams, stack_in, ctx, ctx_plus_emb, x_mask,
                       y_emb, width, dim, prefix, train):
        input_shape = stack_in.shape
        layer_out, _ = theano.scan(self._decoder_block,
                                   sequences=(tparams[self._prepend(prefix, 'W')],
                                              tparams[self._prepend(prefix, 'b')],
                                              tparams[self._prepend(prefix, 'Wg')],
                                              tparams[self._prepend(prefix, 'bg')],
                                              tparams[self._prepend(prefix, 'W_ATT_in')],
                                              tparams[self._prepend(prefix, 'b_ATT_in')],
                                              tparams[self._prepend(prefix, 'W_ATT_out')],
                                              tparams[self._prepend(prefix, 'b_ATT_out')]),
                                   outputs_info=stack_in,
                                   non_sequences=(ctx, ctx_plus_emb, x_mask, y_emb, width, dim,
                                                  input_shape, train))
        return layer_out[-1]

    def _decoder_block(self, W, b, Wg, bg, W_ATT_in, b_ATT_in, W_ATT_out,
                       b_ATT_out, block_in, ctx, ctx_plus_emb, x_mask, y_emb, width, dim,
                       input_shape, train):
        batch_size = input_shape[0]
        # input dropout
        conv_in = self._dropout_layer(block_in, train,
                                      dropout_shape=input_shape)
        # pad left border
        padding = T.zeros((batch_size, width - 1, dim))
        conv_in = T.concatenate((padding, conv_in), axis=1)
        # convolution_block
        state_pre_attn = self._convolution_block(W, b, Wg, bg, conv_in)
        # attention
        state_post_attn = self._attention_block(W_ATT_in, b_ATT_in, W_ATT_out,
                                                b_ATT_out, ctx, ctx_plus_emb, x_mask, y_emb, state_pre_attn)
        state_combined = (state_pre_attn + state_post_attn) / sqrt(2)
        # residual connections
        result = (state_combined + block_in) / sqrt(2)
        return result

    def _init_decoder_stack(self, params, prefix, width, depth, dim, dropout):
        params[self._prepend(prefix, 'W')] = self._normal_weights(
            size=(depth, dim, dim, width),
            scale=sqrt((4.0 * (1.0 - dropout)) / (dim * width)))
        params[self._prepend(prefix, 'b')] = self._zero_weights(
            (depth, dim))
        params[self._prepend(prefix, 'Wg')] = self._normal_weights(
            size=(depth, dim, dim, width),
            scale=sqrt((4.0 * (1.0 - dropout)) / (dim * width)))
        params[self._prepend(prefix, 'bg')] = self._zero_weights(
            (depth, dim))
        params[self._prepend(prefix, 'W_ATT_in')] = self._normal_weights(
            size=(depth, dim, self.config.emb_dim),
            scale=sqrt(1. / dim))
        params[self._prepend(prefix, 'b_ATT_in')] = self._zero_weights(
            (depth, self.config.emb_dim))
        params[self._prepend(prefix, 'W_ATT_out')] = self._normal_weights(
            size=(depth, self.config.emb_dim, dim),
            scale=sqrt(1. / self.config.emb_dim))
        params[self._prepend(prefix, 'b_ATT_out')] = self._zero_weights(
            (depth, dim))

    def _init_decoder(self, params, prefix, dropout):
        arch = self.config.decoder_arch
        self._init_emb_layer(params, self._prepend(prefix, 'w_emb'),
                             self.y_vocab.size())
        in_dim = arch[0][2]
        self._init_ff_layer(params, self._prepend(prefix, 'prj_in'),
                            self.config.emb_dim, in_dim,
                            dropout=dropout)
        dim = None
        for idx, spec in enumerate(arch):
            stack_name = 'stack_' + str(idx)
            depth, width, dim = spec
            if dim != in_dim:
                prj_name = 'stack_prj_in_' + str(idx)
                self._init_ff_layer(params, self._prepend(prefix, prj_name),
                                    in_dim, dim)
            self._init_decoder_stack(params, self._prepend(prefix, stack_name),
                                     width, depth, dim, dropout)
            in_dim = dim
        self._init_ff_layer(params, self._prepend(prefix, 'prj_out_1'),
                            dim, self.config.out_emb_dim)
        self._init_ff_layer(params, self._prepend(prefix, 'prj_out_2'),
                            self.config.out_emb_dim, self.y_vocab.size(),
                            dropout=dropout)

    def _build_model(self, tparams):
        # setup
        x = T.imatrix('x')
        train = T.iscalar()
        x_mask = T.matrix('x_mask')
        y_lshifted = T.imatrix('y_lshifted')
        y_rshifted = T.imatrix('y_rshifted')
        y_mask = T.matrix('y_mask')
        batch_size = y_rshifted.shape[0]
        max_y = y_rshifted.shape[1]
        # encoder
        ctx, ctx_plus_emb = self._encoder(tparams, x, 'enc', train)
        # decoder
        logit = self._decoder(tparams, y_rshifted, ctx, ctx_plus_emb, x_mask,
                              'dec', train)
        # target probs
        n_target_words = logit.shape[2]
        logit_shaped = logit.reshape([batch_size * max_y, n_target_words])
        probs = T.nnet.softmax(logit_shaped)
        # cost
        y_flat = y_lshifted.flatten()
        y_flat_idx = T.arange(batch_size * max_y) * n_target_words + y_flat
        probs = probs.flatten()[y_flat_idx]
        probs = T.switch(T.lt(probs, 1e-8), 1e-8, probs)
        log_probs = -T.log(probs)
        masked_log_probs = log_probs * y_mask.flatten()
        cost = T.mean(masked_log_probs)
        lr = T.scalar('learning_rate', dtype=tconfig.floatX)
        return x, x_mask, y_lshifted, y_rshifted, y_mask, cost, train, lr

    @staticmethod
    def _clip_gradients(grads, limit):
        g2 = 0.
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(T.switch(g2 > (limit ** 2),
                                      (g / T.sqrt(g2)) * limit,
                                      g))
        return new_grads

    def _decoder_step(self, tparams, prev_state, prev_w, ctx, ctx_plus_emb,
                      x_mask, prefix):
        arch = self.config.decoder_arch
        # embed previous word
        prev_w_emb = self._emb_layer(tparams, prev_w.dimshuffle(0, 'x'),
                                     self._prepend(prefix, 'w_emb'))
        prev_w_emb = self._dropout_layer(prev_w_emb, train=0)
        prev_w_emb = T.unbroadcast(prev_w_emb, 1)
        # transform input
        conv_in = self._ff_layer(tparams, prev_w_emb,
                                 self._prepend(prefix, 'prj_in'), self._linear)
        # decode step
        in_dim = arch[0][2]
        next_state = []
        conv_out = None
        for idx, spec in enumerate(arch):
            stack_name = 'stack_' + str(idx)
            depth, width, dim = spec
            if dim != in_dim:
                # transform input for this stack
                prj_name = 'stack_prj_in_' + str(idx)
                conv_in = self._ff_layer(tparams, conv_in,
                                         self._prepend(prefix, prj_name), self._linear)
            # convolution stack
            conv_out, next_s = self._decoder_stack_step(tparams, conv_in,
                                                        prev_state[idx], ctx, ctx_plus_emb, x_mask, prev_w_emb, width,
                                                        self._prepend(prefix, stack_name))
            in_dim = dim
            conv_in = conv_out
            next_state.append(next_s)
        # transform output
        dec_out = self._ff_layer(tparams, conv_out, self._prepend(prefix,
                                                                  'prj_out_1'), self._linear)
        # dropout
        dec_out = self._dropout_layer(dec_out, train=0)
        # return logit
        logit = self._ff_layer(tparams, dec_out, self._prepend(prefix,
                                                               'prj_out_2'), self._linear)
        return logit, next_state

    def _decoder_stack_step(self, tparams, stack_in, prev_state, ctx,
                            ctx_plus_emb, x_mask, prev_w_emb, width, prefix):
        prev_state = prev_state.dimshuffle(1, 0, 2, 3)
        input_shape = stack_in.shape
        outs, _ = theano.scan(self._decoder_block_step,
                              sequences=(tparams[self._prepend(prefix, 'W')],
                                         tparams[self._prepend(prefix, 'b')],
                                         tparams[self._prepend(prefix, 'Wg')],
                                         tparams[self._prepend(prefix, 'bg')],
                                         tparams[self._prepend(prefix, 'W_ATT_in')],
                                         tparams[self._prepend(prefix, 'b_ATT_in')],
                                         tparams[self._prepend(prefix, 'W_ATT_out')],
                                         tparams[self._prepend(prefix, 'b_ATT_out')],
                                         prev_state),
                              outputs_info=(stack_in, None),
                              non_sequences=(ctx, ctx_plus_emb, x_mask, prev_w_emb, width, input_shape))
        layers_out = outs[0]
        next_state = outs[1]
        stack_out = layers_out[-1]
        next_state = next_state.dimshuffle(1, 0, 2, 3)
        return stack_out, next_state

    def _decoder_block_step(self, W, b, Wg, bg, W_ATT_in, b_ATT_in, W_ATT_out,
                            b_ATT_out, prev_state, block_in, ctx, ctx_plus_emb, x_mask, prev_w_emb,
                            width, input_shape):
        # input dropout
        conv_in = self._dropout_layer(block_in, train=0,
                                      dropout_shape=input_shape)
        # prepend previous inputs
        conv_in = T.concatenate((prev_state, conv_in), axis=1)
        next_state = conv_in[:, 1:width, :]
        # convolution_block
        state_pre_attn = self._convolution_block(W, b, Wg, bg, conv_in)
        # attention
        state_post_attn = self._attention_block(W_ATT_in, b_ATT_in, W_ATT_out,
                                                b_ATT_out, ctx, ctx_plus_emb, x_mask, prev_w_emb, state_pre_attn)
        state_combined = (state_pre_attn + state_post_attn) / sqrt(2)
        # residual connections
        result = (state_combined + block_in) / sqrt(2)
        return result, next_state

    def _build_sampler(self, tparams):
        # input to encoder
        x = T.imatrix('x')
        # encode context
        ctx, ctx_plus_emb = self._encoder(tparams, x, 'enc', train=0)
        # initializer function
        f_context = theano.function([x], [ctx, ctx_plus_emb])

        def f_init(x_):
            ctx_, ctx_plus_emb_ = f_context(x_)
            init_state = [np.zeros([1, a[0], a[1] - 1, a[2]], dtype=np.float32)
                          for a in self.config.decoder_arch]
            return init_state, ctx_, ctx_plus_emb_

        # inputs to decoder
        prev_state = [T.tensor4() for _ in self.config.decoder_arch]
        ctx = T.tensor3('ctx')
        ctx_plus_emb = T.tensor3('ctx_plus_emb')
        x_mask = T.ones((1, ctx.shape[1]))
        prev_w = T.ivector('prev_w')
        batch_size = prev_state[0].shape[0]
        # decoder step
        logit, next_state = self._decoder_step(tparams, prev_state, prev_w,
                                               ctx, ctx_plus_emb, x_mask, 'dec')
        # next word and state
        n_target_words = logit.shape[2]
        logit_shaped = logit.reshape([batch_size, n_target_words])
        next_probs = T.nnet.softmax(logit_shaped)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)
        # next word/state function
        f_next_T = theano.function([*prev_state, ctx, ctx_plus_emb, prev_w],
                                   [next_probs, next_sample, *next_state])

        def f_next(prev_state_, ctx_, ctx_plus_emb_, prev_w_):
            result = f_next_T(*prev_state_, ctx_, ctx_plus_emb_, prev_w_)
            return result[0], result[1], result[2:]

        return f_init, f_next

    def _gen_sample(self, f_init, f_next, x, max_words, stochastic=False):
        result = []
        cost = 0
        state, ctx, ctx_plus_emb = f_init([x])
        bos = self.y_vocab.lookup(Vocab.SENT_START)
        eos = self.y_vocab.lookup(Vocab.SENT_END)
        w = np.array([bos]).astype('int32')
        for i in range(max_words):
            probs, sample, new_state = f_next(state, ctx, ctx_plus_emb, w)
            if stochastic:
                next_w = sample[0]
            else:
                next_w = np.argmax(probs)
            cost -= np.math.log(probs[0, next_w])
            result.append(next_w)
            if next_w == eos:
                break
            w = np.array([next_w]).astype('int32')
            state = new_state
        return result, cost

    def _gen_nbest(self, f_init, f_next, x, n, max_words):
        num_active = 1
        num_complete = 0
        complete_theories = []
        complete_costs = []
        partial_theories = [[]]
        partial_costs = np.zeros((1,)).astype('float32')
        state, ctx0, ctx_plus_emb0 = f_init([x])
        bos = self.y_vocab.lookup(Vocab.SENT_START)
        eos = self.y_vocab.lookup(Vocab.SENT_END)
        w = np.array([bos]).astype('int32')
        for i in range(max_words):
            ctx = np.tile(ctx0, (num_active, 1, 1))
            ctx_plus_emb = np.tile(ctx_plus_emb0, (num_active, 1, 1))
            probs, sample, new_state = f_next(state, ctx, ctx_plus_emb, w)
            cand_costs = partial_costs[:, np.newaxis] - np.log(probs)
            cand_costs_flat = cand_costs.flatten()
            ranks_flat = cand_costs_flat.argsort()[:n - num_complete]
            voc_size = self.y_vocab.size()
            sent_indices = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_costs_flat[ranks_flat]
            new_theories = []
            new_costs = np.zeros(n - num_complete).astype('float32')
            new_states = [[] for _ in range(self.num_decoder_stacks)]
            for idx, [si, wi] in enumerate(zip(sent_indices, word_indices)):
                new_theories.append(partial_theories[si] + [wi])
                new_costs[idx] = copy.copy(costs[idx])
                self._append_state(new_states, new_state, si, copy_state=True)
            new_active = 0
            partial_theories = []
            partial_costs = []
            partial_states = [[] for _ in range(self.num_decoder_stacks)]
            for idx in range(len(new_theories)):
                if new_theories[idx][-1] == eos:
                    complete_theories.append(new_theories[idx])
                    complete_costs.append(new_costs[idx])
                    num_complete += 1
                else:
                    new_active += 1
                    partial_theories.append(new_theories[idx])
                    partial_costs.append(new_costs[idx])
                    self._append_state(partial_states, new_states, idx)
            partial_costs = np.array(partial_costs)
            num_active = new_active
            if num_active < 1:
                break
            if num_complete >= n:
                break
            w = np.array([sent[-1] for sent
                          in partial_theories]).astype('int32')
            state = [np.array(s) for s in partial_states]
        if num_active > 0:
            for idx in range(num_active):
                complete_theories.append(partial_theories[idx])
                complete_costs.append(partial_costs[idx])
        return complete_theories, complete_costs

    def _append_state(self, new_states, old_states, state_idx,
                      copy_state=False):
        for i in range(self.num_decoder_stacks):
            state = old_states[i][state_idx] if not copy_state else \
                copy.copy(old_states[i][state_idx])
            new_states[i].append(state)

    def _build_batch_sampler(self, tparams):
        x = T.imatrix('x')
        x_mask = T.matrix('x_mask')
        n = T.iscalar("n")
        ctx, ctx_plus_emb = self._encoder(tparams, x, 'enc', train=0)
        state = [T.zeros([x.shape[0], a[0], a[1] - 1, a[2]])
                 for a in self.config.decoder_arch]
        w = T.zeros([x.shape[0]], dtype='int64') + self.y_vocab.lookup(Vocab.SENT_START)
        decoder_param_keys, decoder_param_vals = self.get_decoder_shared_vars()
        non_seqs = [ctx, ctx_plus_emb, x_mask]
        non_seqs.extend(decoder_param_vals)

        def f_step(*args):
            k = self.num_decoder_stacks
            prev_state = list(args[:k])
            prev_w = args[k]
            ctx_ = args[k + 1]
            ctx_plus_emb_ = args[k + 2]
            x_mask_ = args[k + 3]
            param_vals = args[k + 4:]
            dec_params = {}
            for k, v in zip(decoder_param_keys, param_vals):
                dec_params[k] = v
            batch_size = prev_state[0].shape[0]
            logit, next_state = self._decoder_step(dec_params, prev_state, prev_w,
                                                   ctx_, ctx_plus_emb_, x_mask_, 'dec')
            n_target_words = logit.shape[2]
            logit_shaped = logit.reshape([batch_size, n_target_words])
            next_w = T.argmax(logit_shaped, axis=1)
            return next_state + [next_w]

        result, updates = theano.scan(fn=f_step,
                                      outputs_info=state + [w],
                                      non_sequences=non_seqs,
                                      n_steps=n,
                                      strict=True)
        final_result = result[-1].dimshuffle(1, 0)
        f_sample = theano.function([x, x_mask, n], final_result, updates=updates)
        return f_sample

    def get_decoder_shared_vars(self):
        keys = []
        vals = []
        for k, v in self.tparams.items():
            if k.startswith('dec_'):
                keys.append(k)
                vals.append(v)
        return keys, vals
