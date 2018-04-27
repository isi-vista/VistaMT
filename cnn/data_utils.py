import numpy as np

from cnn.vocab import Vocab


def batch_generator(xpath, ypath, x_vocab, y_vocab, batch_size, max_words):
    seqs_x = []
    seqs_y = []
    with open(xpath, 'r', encoding='utf8') as fx, \
            open(ypath, 'r', encoding='utf8') as fy:
        for xline, yline in zip(fx, fy):
            xline = xline.rstrip()
            yline = yline.rstrip()
            xwords = []
            ywords = []
            xwords.extend(xline.split())
            xwords.append(Vocab.SENT_END)
            ywords.append(Vocab.SENT_START)
            ywords.extend(yline.split())
            ywords.append(Vocab.SENT_END)
            if len(xwords) > max_words or len(ywords) > max_words:
                continue
            xindexes = [x_vocab.lookup(w) for w in xwords]
            yindexes = [y_vocab.lookup(w) for w in ywords]
            seqs_x.append(xindexes)
            seqs_y.append(yindexes)
            if len(seqs_y) == batch_size:
                yield seqs_x, seqs_y
                seqs_x = []
                seqs_y = []
        if seqs_y:
            yield seqs_x, seqs_y


def prepare_data(seqs_x, seqs_y):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(seqs_y)
    maxlen_x = np.max(lengths_x)
    maxlen_y = np.max(lengths_y)
    x = np.zeros((n_samples, maxlen_x)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    y_lshifted = np.zeros((n_samples, maxlen_y - 1)).astype('int32')
    y_rshifted = np.zeros((n_samples, maxlen_y - 1)).astype('int32')
    y_mask = np.zeros((n_samples, maxlen_y - 1)).astype('float32')
    for idx, (s_x, s_y) in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1
        y_lshifted[idx, :lengths_y[idx] - 1] = s_y[1:lengths_y[idx]]
        y_rshifted[idx, :lengths_y[idx] - 1] = s_y[:lengths_y[idx] - 1]
        y_mask[idx, :lengths_y[idx] - 1] = 1
    return x, x_mask, y_lshifted, y_rshifted, y_mask


# src side batching, for greedy batch predict
def x_batch_generator(xpath, x_vocab, batch_size, max_words):
    seqs_x = []
    with open(xpath, 'r', encoding='utf8') as fx:
        for xline in fx:
            xline = xline.rstrip()
            xwords = []
            xwords.extend(xline.split())
            xwords.append(Vocab.SENT_END)
            if len(xwords) > max_words:
                continue
            xindexes = [x_vocab.lookup(w) for w in xwords]
            seqs_x.append(xindexes)
            if len(seqs_x) == batch_size:
                yield seqs_x
                seqs_x = []
        if seqs_x:
            yield seqs_x


# src side batching, for greedy batch predict
def x_prepare_data(seqs_x):
    lengths_x = [len(s) for s in seqs_x]
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros((n_samples, maxlen_x)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1
    return x, x_mask
