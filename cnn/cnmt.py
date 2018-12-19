from math import sqrt
import tensorflow as tf
import numpy as np

from cnn.vocab import Vocab
from cnn.layers import embedding, linear, conv_glu


class CNMT_1:
    def __init__(self, config, x_vocab, y_vocab):
        #tf.set_random_seed(1234)
        self.learning_rate = 0.0002
        self.config = config
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.dropout_rate = self.config.dropout_rate
        num_decoder_layers = 0
        for a in self.config.decoder_arch:
            num_decoder_layers += a[0]
        self.num_decoder_layers = num_decoder_layers
        self.session = tf.Session()
        self.f_train = self._build_f_train()
        self.f_cost = self._build_f_cost()
        self.f_predict = self._build_f_predict()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

    def load_params(self, filename):
        self.saver.restore(self.session, filename)

    def save_params(self, filename):
        self.saver.save(self.session, filename)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def train(self, x, x_mask, y_lshifted, y_rshifted, y_mask):
        return self.f_train(x, x_mask, y_lshifted, y_rshifted, y_mask)

    def predict(self, x, x_mask, maxlen, beam_size):
        return self.f_predict(x, x_mask, maxlen, beam_size)

    def get_cost(self, x, x_mask, y_lshifted, y_rshifted, y_mask):
        return self.f_cost(x, x_mask, y_lshifted, y_rshifted, y_mask)

    def _encoder(self, name, x, x_mask, training):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            arch = self.config.encoder_arch
            emb_dim = self.config.emb_dim
            # embed source
            x_emb = embedding('w_emb', x, self.x_vocab.size(), emb_dim)
            # mask unused inputs
            x_emb = x_emb * tf.expand_dims(x_mask, axis=2)
            # dropout
            x_emb = tf.layers.dropout(x_emb, rate=self.dropout_rate, training=training)
            # transform input for first hidden layer
            curr_dim = arch[0][2]
            h_enc = linear('prj_in', x_emb, emb_dim, curr_dim, self.dropout_rate)
            # convolutions
            for idx, spec in enumerate(arch):
                depth, width, dim = spec
                if dim != curr_dim:
                    # transform input for this stack
                    prj_name = 'stack_prj_in_' + str(idx)
                    h_enc = linear(prj_name, h_enc, curr_dim, dim)
                # convolution stack
                stack_name = 'stack_' + str(idx)
                h_enc = self._encoder_stack(stack_name, h_enc, x_mask, width, depth, dim, training)
                curr_dim = dim
            # transform output
            ctx = linear('prj_out', h_enc, curr_dim, emb_dim)
            # scale gradients
            scale_factor = 1. / (2 * self.num_decoder_layers)
            ctx = (scale_factor * ctx) + ((1 - scale_factor) * tf.stop_gradient(ctx))
            # context plus word embedding
            ctx_plus_emb = (ctx + x_emb) / sqrt(2)
            return ctx, ctx_plus_emb

    def _encoder_stack(self, name, stack_in, x_mask, width, depth, dim, training):
        with tf.variable_scope(name):
            x = stack_in
            # stacked encoder blocks
            for i in range(depth):
                block_name = 'blk_' + str(i)
                x = self._encoder_block(block_name, x, x_mask, width, dim, training)
            return x

    def _encoder_block(self, name, block_in, x_mask, width, dim, training):
        with tf.variable_scope(name):
            # dropout
            conv_in = tf.layers.dropout(block_in, rate=self.dropout_rate, training=training)
            # mask unused inputs
            conv_in = conv_in * tf.expand_dims(x_mask, axis=2)
            # convolution followed by gated linear units
            conv_out = conv_glu('conv_glu', conv_in, width, dim, 'SAME', self.dropout_rate)
            # residual connections
            result = (conv_out + block_in) / sqrt(2)
            return result

    def _attention_block(self, name, ctx, ctx_plus_emb, x_mask, prev_w_emb, state_pre_attn, dec_dim):
        with tf.variable_scope(name):
            emb_dim = self.config.emb_dim
            x_len = tf.reduce_sum(x_mask, axis=1)
            # project state to embedding dimension
            state_attn = linear('prj_attn_in', state_pre_attn, dec_dim, emb_dim)
            # add previous word
            state_attn = (state_attn + prev_w_emb) / sqrt(2)
            # calculate weight for each source position
            alpha = tf.einsum('bye,bxe->byx', state_attn, ctx)
            inf_mask = tf.where(tf.equal(x_mask, 1.), tf.zeros_like(x_mask, dtype=tf.float32),
                                tf.ones_like(x_mask, dtype=tf.float32) * np.NINF)
            alpha = alpha + tf.expand_dims(inf_mask, axis=1)
            alpha = alpha - tf.reduce_max(alpha, axis=2, keepdims=True)
            alpha = tf.exp(alpha)
            alpha = alpha * tf.expand_dims(x_mask, axis=1)
            alpha = alpha / tf.reduce_sum(alpha, axis=2, keepdims=True)
            # weighted average of source contributions
            ctx_contrib = tf.expand_dims(alpha, axis=3) * tf.expand_dims(ctx_plus_emb, axis=1)
            ctx_contrib = tf.reduce_sum(ctx_contrib, axis=2)
            # rescale and control for variance
            ctx_scale_factor = x_len / tf.sqrt(x_len)
            ctx_scale_factor = tf.expand_dims(tf.expand_dims(ctx_scale_factor, axis=1), axis=2)
            ctx_attn = ctx_contrib * ctx_scale_factor
            # project back to decoder state dimension
            ctx_attn = linear('prj_attn_out', ctx_attn, emb_dim, dec_dim)
            return ctx_attn

    def _decoder(self, name, y_in, ctx, ctx_plus_emb, x_mask, step_mode, prev_state, training):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if step_mode:
                y_in = tf.expand_dims(y_in, axis=1)
            next_state = [] if step_mode else None
            arch = self.config.decoder_arch
            emb_dim = self.config.emb_dim
            # embed target
            y_emb = embedding('w_emb', y_in, self.y_vocab.size(), emb_dim)
            # dropout
            y_emb = tf.layers.dropout(y_emb, rate=self.dropout_rate, training=training)
            # transform input for first hidden layer
            curr_dim = arch[0][2]
            h_dec = linear('prj_in', y_emb, emb_dim, curr_dim, self.dropout_rate)
            # convolutions
            for idx, spec in enumerate(arch):
                depth, width, dim = spec
                if dim != curr_dim:
                    # transform input for this stack
                    prj_name = 'stack_prj_in_' + str(idx)
                    h_dec = linear(prj_name, h_dec, curr_dim, dim)
                # convolution stack
                stack_name = 'stack_' + str(idx)
                stack_prev_state = prev_state[idx] if step_mode else None
                h_dec, stack_next_state = self._decoder_stack(stack_name, h_dec, ctx, ctx_plus_emb, x_mask, y_emb,
                                                              width, depth, dim, step_mode, stack_prev_state, training)
                if step_mode:
                    next_state.append(stack_next_state)
                curr_dim = dim
            next_state = tuple(next_state) if next_state is not None else None
            # transform to output embedding dimension
            dec_out = linear('prj_out_1', h_dec, curr_dim, self.config.out_emb_dim)
            # dropout
            dec_out = tf.layers.dropout(dec_out, rate=self.dropout_rate, training=training)
            # output layer
            logits = linear('prj_out_2', dec_out, self.config.out_emb_dim, self.y_vocab.size(), self.dropout_rate)
            return logits, next_state

    def _decoder_stack(self, name, stack_in, ctx, ctx_plus_emb, x_mask, y_emb, width, depth, dim, step_mode,
                       prev_state, training):
        with tf.variable_scope(name):
            next_state = [] if step_mode else None
            y = stack_in
            # stacked decoder blocks
            for i in range(depth):
                block_name = 'blk_' + str(i)
                block_prev_state = prev_state[i] if step_mode else None
                y, block_next_state = self._decoder_block(block_name, y, ctx, ctx_plus_emb, x_mask, y_emb, width, dim,
                                                          step_mode, block_prev_state, training)
                if step_mode:
                    next_state.append(block_next_state)
            next_state = tuple(next_state) if next_state is not None else None
            return y, next_state

    def _decoder_block(self, name, block_in, ctx, ctx_plus_emb, x_mask, y_emb, width, dim, step_mode, prev_state,
                       training):
        with tf.variable_scope(name):
            batch_size = tf.shape(block_in)[0]
            # dropout
            conv_in = tf.layers.dropout(block_in, rate=self.dropout_rate, training=training)
            # prepend previous state (zeros if not in step mode)
            if not step_mode:
                prev_state = tf.zeros((batch_size, width - 1, dim))
            conv_in = tf.concat((prev_state, conv_in), axis=1)
            # prepare next state (none if not in step mode)
            next_state = None if not step_mode else conv_in[:, 1:width, :]
            # convolution_block
            state_pre_attn = conv_glu('conv_glu', conv_in, width, dim, 'VALID', self.dropout_rate)
            # attention
            ctx_attn = self._attention_block('attn', ctx, ctx_plus_emb, x_mask, y_emb, state_pre_attn, dim)
            state_combined = (state_pre_attn + ctx_attn) / sqrt(2)
            # residual connections
            result = (state_combined + block_in) / sqrt(2)
            return result, next_state

    def _beam_search(self, ctx, ctx_plus_emb, x_mask, maxlen, beam_size):
        batch_size = tf.shape(ctx)[0]
        f_min = np.finfo(np.float32).min

        # initial states for loop variables
        i = tf.constant(0)
        init_state = self._beam_initial_state(batch_size)
        init_ys = tf.zeros(dtype=tf.int32, shape=[batch_size]) + self.y_vocab.lookup(Vocab.SENT_START)
        init_cost = tf.concat([tf.zeros([1]), tf.zeros([beam_size-1]) + f_min], axis=0)
        init_cost = tf.tile(init_cost, multiples=[batch_size / beam_size])
        words_out = tf.TensorArray(dtype=tf.int32, size=maxlen, clear_after_read=True)
        backtrace = tf.TensorArray(dtype=tf.int32, size=maxlen, clear_after_read=True)
        init_loop_vars = [i, init_state, init_ys, init_cost, words_out, backtrace]

        # for completed sentences, transition cost is 0 if eos, min float for all other words
        eos_log_probs = tf.constant([[0.] + ([f_min] * (self.y_vocab.size() - 1))], dtype=tf.float32)
        eos_log_probs = tf.tile(eos_log_probs, multiples=[batch_size, 1])

        def cond(_i, _prev_state, _prev_w, _prev_cost, _words_out, _backtrace):
            return tf.logical_and(tf.less(_i, maxlen), tf.reduce_any(tf.not_equal(_prev_w, 0)))

        def body(_i, _prev_state, _prev_w, _prev_cost, _words_out, _backtrace):
            logits, state = self._decoder('dec', _prev_w, ctx, ctx_plus_emb, x_mask, True, _prev_state, False)
            logits = tf.squeeze(logits, axis=1)
            log_probs = tf.nn.log_softmax(logits)
            log_probs = tf.where(tf.equal(_prev_w, 0), eos_log_probs, log_probs)
            all_costs = log_probs + tf.expand_dims(_prev_cost, axis=1)
            all_costs = tf.reshape(all_costs, shape=[-1, self.y_vocab.size() * beam_size])
            values, _indices = tf.nn.top_k(all_costs, k=beam_size)
            new_cost = tf.reshape(values, shape=[batch_size])
            offsets = tf.range(start=0, delta=beam_size, limit=batch_size, dtype=tf.int32)
            offsets = tf.expand_dims(offsets, axis=1)
            predecessors = tf.div(_indices, self.y_vocab.size()) + offsets
            predecessors = tf.reshape(predecessors, shape=[batch_size])
            new_state = self._beam_update_state(state, predecessors)
            new_w = _indices % self.y_vocab.size()
            new_w = tf.reshape(new_w, shape=[batch_size])
            new_cost = tf.where(tf.equal(new_w, 0), tf.abs(new_cost), new_cost)
            _words_out = _words_out.write(_i, value=new_w)
            _backtrace = _backtrace.write(_i, value=predecessors)
            return _i + 1, new_state, new_w, new_cost, _words_out, _backtrace

        loop_result = tf.while_loop(cond=cond, body=body, loop_vars=init_loop_vars, back_prop=False)
        i, _, _, cost, words_out, backtrace = loop_result

        indices = tf.range(0, i)
        words_out = words_out.gather(indices)
        backtrace = backtrace.gather(indices)
        cost = tf.abs(cost)
        return words_out, backtrace, cost

    def _beam_initial_state(self, batch_size):
        # setup initial decoder state
        state = []
        for stack in self.config.decoder_arch:
            stack_state = []
            num_blocks = stack[0]
            block_width = stack[1]
            block_dim = stack[2]
            for block in range(num_blocks):
                block_state = tf.zeros([batch_size, block_width - 1, block_dim])
                stack_state.append(block_state)
            stack_state = tuple(stack_state)
            state.append(stack_state)
        state = tuple(state)
        return state

    def _beam_update_state(self, state, predecessors):
        new_state = []
        stack_num = 0
        for stack in self.config.decoder_arch:
            stack_state = []
            num_blocks = stack[0]
            for block in range(num_blocks):
                old_state = state[stack_num][block]
                updated_state = tf.gather(old_state, predecessors)
                stack_state.append(updated_state)
            stack_state = tuple(stack_state)
            new_state.append(stack_state)
            stack_num += 1
        new_state = tuple(new_state)
        return new_state

    @staticmethod
    def _recover_sentences_from_beam(words_out, backtrace):
        result = []
        for sent_idx in range(words_out.shape[1]):
            sentence = []
            pos = backtrace.shape[0] - 1
            idx = sent_idx
            while pos >= 0:
                w = words_out[pos, idx]
                idx = backtrace[pos, idx]
                pos -= 1
                sentence.append(w)
            sentence.reverse()
            sentence = np.trim_zeros(sentence, 'b')
            sentence.append(0)
            result.append(sentence)
        return result

    def _build_f_train(self):
        # graph inputs
        _x = tf.placeholder(tf.int32, shape=(None, None))
        _x_mask = tf.placeholder(tf.float32, shape=(None, None))
        _y_lshifted = tf.placeholder(tf.int32, shape=(None, None))
        _y_rshifted = tf.placeholder(tf.int32, shape=(None, None))
        _y_mask = tf.placeholder(tf.float32, shape=(None, None))
        _learning_rate = tf.placeholder(tf.float32)
        # encoder
        ctx, ctx_plus_emb = self._encoder('enc', _x, _x_mask, True)
        # decoder
        logits, _ = self._decoder('dec', _y_rshifted, ctx, ctx_plus_emb, _x_mask, False, [], True)
        # loss
        loss = tf.losses.sparse_softmax_cross_entropy(_y_lshifted, logits, weights=_y_mask)
        # optimizer
        train_op = tf.train.AdamOptimizer(_learning_rate).minimize(loss)

        def train_fn(x, x_mask, y_lshifted, y_rshifted, y_mask):
            result, _ = self.session.run([loss, train_op], feed_dict={_x: x, _x_mask: x_mask, _y_lshifted: y_lshifted,
                                                                      _y_rshifted: y_rshifted, _y_mask: y_mask,
                                                                      _learning_rate: self.learning_rate})
            return result

        return train_fn

    def _build_f_cost(self):
        # graph inputs
        _x = tf.placeholder(tf.int32, shape=(None, None))
        _x_mask = tf.placeholder(tf.float32, shape=(None, None))
        _y_lshifted = tf.placeholder(tf.int32, shape=(None, None))
        _y_rshifted = tf.placeholder(tf.int32, shape=(None, None))
        _y_mask = tf.placeholder(tf.float32, shape=(None, None))
        # encoder
        ctx, ctx_plus_emb = self._encoder('enc', _x, _x_mask, False)
        # decoder
        logits, _ = self._decoder('dec', _y_rshifted, ctx, ctx_plus_emb, _x_mask, False, [], False)
        # loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y_lshifted, logits=logits)
        cross_entropy = cross_entropy * _y_mask
        cost = tf.reduce_sum(cross_entropy) / tf.reduce_sum(_y_mask)

        def cost_fn(x, x_mask, y_lshifted, y_rshifted, y_mask):
            result = self.session.run(cost, feed_dict={_x: x, _x_mask: x_mask, _y_lshifted: y_lshifted,
                                                       _y_rshifted: y_rshifted, _y_mask: y_mask})
            return result

        return cost_fn

    def _build_f_predict(self):
        # graph inputs
        _x = tf.placeholder(tf.int32, shape=(None, None))
        _x_mask = tf.placeholder(tf.float32, shape=(None, None))
        _maxlen = tf.placeholder(tf.int32)
        _beam_size = tf.placeholder(tf.int32)
        # encoder
        ctx, ctx_plus_emb = self._encoder('enc', _x, _x_mask, False)
        # beam search
        _words_out, _backtrace, _cost = self._beam_search(ctx, ctx_plus_emb, _x_mask, _maxlen, _beam_size)

        def get_best_theories(cost, beam_size):
            batch_size = cost.shape[0] // beam_size
            result = []
            for batch in range(batch_size):
                batch_start = batch * beam_size
                batch_end = batch_start + beam_size
                result.append(batch_start + np.argmin(cost[batch_start:batch_end]))
            return result

        def predict(x, x_mask, maxlen, beam_size):
            x_rep = np.repeat(x, repeats=beam_size, axis=0)
            x_mask_rep = np.repeat(x_mask, repeats=beam_size, axis=0)
            words_out, backtrace, cost = self.session.run([_words_out, _backtrace, _cost],
                                                          feed_dict={_x: x_rep, _x_mask: x_mask_rep, _maxlen: maxlen,
                                                                     _beam_size: beam_size})
            sentences = self._recover_sentences_from_beam(words_out, backtrace)
            sentence_lengths = [len(s) for s in sentences]
            norm_cost = cost / sentence_lengths
            result = []
            for idx in get_best_theories(norm_cost, beam_size):
                result.append(sentences[idx])
            return result

        return predict
