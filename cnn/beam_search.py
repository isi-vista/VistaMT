import numpy as np
import tensorflow as tf

from cnn.vocab import Vocab


class BeamSearch:
    def __init__(self, encoder, decoder, y_vocab):
        self.encoder = encoder
        self.decoder = decoder
        self.y_vocab = y_vocab

    def predict(self, x, x_mask, maxlen, beam_size):
        x_rep = np.repeat(x, repeats=beam_size, axis=0)
        x_mask_rep = np.repeat(x_mask, repeats=beam_size, axis=0)
        words_out, backtrace, cost = self.forward_pass(x_rep, x_mask_rep, maxlen, beam_size)
        sentences = self.recover_sentences_from_beam(words_out, backtrace)
        sentence_lengths = [len(s) for s in sentences]
        norm_cost = cost / sentence_lengths
        result = []
        for idx in self._get_best_theories(norm_cost, beam_size):
            result.append(sentences[idx])
        return result

    def predict_n(self, x, x_mask, maxlen, beam_size):
        x_rep = np.repeat(x, repeats=beam_size, axis=0)
        x_mask_rep = np.repeat(x_mask, repeats=beam_size, axis=0)
        words_out, backtrace, cost = self.forward_pass(x_rep, x_mask_rep, maxlen, beam_size)
        sentences = self.recover_sentences_from_beam(words_out, backtrace)
        sentence_lengths = [len(s) for s in sentences]
        norm_cost = cost.numpy() / sentence_lengths
        batch_ordered_beam_sent_indices = self._get_ranked_theories(norm_cost, beam_size)
        result = []  # batch, beam, (sent, cost)
        for batch_sent_idx, beam_sent_indices in enumerate(batch_ordered_beam_sent_indices):
            beam_sents = []
            for beam_idx, sent_idx in enumerate(beam_sent_indices):
                beam_sents.append((sentences[sent_idx], norm_cost[sent_idx]))
            result.append(beam_sents)
        return result

    @tf.function(input_signature=(tf.TensorSpec((None, None), dtype=tf.int32),
                                  tf.TensorSpec((None, None), dtype=tf.float32),
                                  tf.TensorSpec((), dtype=tf.int32),
                                  tf.TensorSpec((), dtype=tf.int32)))
    def forward_pass(self, x, x_mask, maxlen, beam_size):
        batch_size = tf.shape(x)[0]
        f_min = np.finfo(np.float32).min

        i = tf.constant(0)
        init_state = self._beam_initial_state(batch_size)
        init_ys = tf.zeros(dtype=tf.int32, shape=[batch_size]) + self.y_vocab.lookup(
            Vocab.SENT_START)
        init_cost = tf.concat([tf.zeros([1]), tf.zeros([beam_size - 1]) + f_min], axis=0)
        init_cost = tf.tile(init_cost, multiples=[batch_size / beam_size])
        words_out = tf.TensorArray(dtype=tf.int32, size=maxlen, clear_after_read=True)
        backtrace = tf.TensorArray(dtype=tf.int32, size=maxlen, clear_after_read=True)
        init_loop_vars = [i, init_state, init_ys, init_cost, words_out, backtrace]

        # for completed sentences, transition cost is 0 if eos, min float for all other words
        eos_log_probs = tf.constant([[0.] + ([f_min] * (self.y_vocab.size() - 1))],
                                    dtype=tf.float32)
        eos_log_probs = tf.tile(eos_log_probs, multiples=[batch_size, 1])

        ctx, ctx_plus_emb = self.encoder([x, x_mask], training=False)

        def cond(_i, _prev_state, _prev_w, _prev_cost, _words_out, _backtrace):
            return tf.logical_and(tf.less(_i, maxlen), tf.reduce_any(tf.not_equal(_prev_w, 0)))

        def body(_i, _prev_state, _prev_w, _prev_cost, _words_out, _backtrace):
            logits, state = self.decoder([_prev_w, ctx, ctx_plus_emb, x_mask, _prev_state],
                                         step_mode=True, training=False)
            logits = tf.squeeze(logits, axis=1)
            log_probs = tf.nn.log_softmax(logits)
            prev_w = tf.expand_dims(_prev_w, axis=1)
            log_probs = tf.where(tf.equal(prev_w, 0), eos_log_probs, log_probs)
            all_costs = log_probs + tf.expand_dims(_prev_cost, axis=1)
            all_costs = tf.reshape(all_costs, shape=[-1, self.y_vocab.size() * beam_size])
            values, _indices = tf.nn.top_k(all_costs, k=beam_size)
            new_cost = tf.reshape(values, shape=[batch_size])
            offsets = tf.range(start=0, delta=beam_size, limit=batch_size, dtype=tf.int32)
            offsets = tf.expand_dims(offsets, axis=1)
            predecessors = tf.math.floordiv(_indices, self.y_vocab.size()) + offsets
            predecessors = tf.reshape(predecessors, shape=[batch_size])
            new_state = self._beam_update_state(state, predecessors)
            new_w = _indices % self.y_vocab.size()
            new_w = tf.reshape(new_w, shape=[batch_size])
            new_cost = tf.where(tf.equal(new_w, 0), tf.abs(new_cost), new_cost)
            _words_out = _words_out.write(_i, value=new_w)
            _backtrace = _backtrace.write(_i, value=predecessors)
            return _i + 1, new_state, new_w, new_cost, _words_out, _backtrace

        loop_result = tf.while_loop(cond=cond, body=body, loop_vars=init_loop_vars)
        i, _, _, cost, words_out, backtrace = loop_result

        indices = tf.range(0, i)
        words_out = words_out.gather(indices)
        backtrace = backtrace.gather(indices)
        cost = tf.abs(cost)
        return words_out, backtrace, cost

    @staticmethod
    def recover_sentences_from_beam(words_out, backtrace):
        result = []
        for sent_idx in range(words_out.shape[1]):
            sentence = []
            pos = backtrace.shape[0] - 1
            idx = sent_idx
            while pos >= 0:
                w = words_out[pos, idx]
                idx = backtrace[pos, idx]
                pos -= 1
                sentence.append(w.numpy())
            sentence.reverse()
            sentence = np.trim_zeros(sentence, 'b')
            sentence.append(0)
            result.append(sentence)
        return result

    def _beam_initial_state(self, batch_size):
        state = []
        for stack in self.decoder.architecture:
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
        for stack in self.decoder.architecture:
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
    def _get_best_theories(cost, beam_size):
        batch_size = cost.shape[0] // beam_size
        result = []
        for batch in range(batch_size):
            batch_start = batch * beam_size
            batch_end = batch_start + beam_size
            result.append(batch_start + np.argmin(cost[batch_start:batch_end]))
        return result

    @staticmethod
    def _get_ranked_theories(cost, beam_size):
        batch_size = cost.shape[0] // beam_size
        result = np.argsort(cost.reshape(batch_size, beam_size))
        result += np.expand_dims(np.arange(0, batch_size * beam_size, beam_size), 1)
        return result
