import logging

import numpy as np

from cnn.vocab import Vocab

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class XYDataset:
    def __init__(self, xpath, ypath, x_vocab, y_vocab, max_words_per_sentence=80,
                 max_words_per_batch=4000, max_sentences_per_batch=200, random_state=None):
        self.xpath = xpath
        self.ypath = ypath
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.max_words = max_words_per_sentence
        self.random_state = random_state
        self.all_x, self.all_y = self._load(xpath, ypath)
        self.batches = self._make_batches(max_words_per_batch, max_sentences_per_batch)
        log.info('loaded {} examples from {}, {}'.format(self.size(), xpath, ypath))

    def __call__(self):
        indices = np.arange(len(self.batches))
        if self.random_state:
            self.random_state.shuffle(indices)
        for idx in indices:
            batch_start, batch_end = self.batches[idx]
            yield self._prepare_data(batch_start, batch_end)

    def size(self):
        return len(self.all_x)

    def _load(self, xpath, ypath):
        all_x = []
        all_x_lengths = []
        all_y = []
        all_y_lengths = []
        with open(xpath, encoding='utf8') as fx, open(ypath, encoding='utf8') as fy:
            for xline, yline in zip(fx, fy):
                xwords = xline.rstrip().split()
                ywords = yline.rstrip().split()
                if len(xwords) > self.max_words or len(ywords) > self.max_words:
                    continue
                xwords.append(Vocab.SENT_END)
                ywords.insert(0, Vocab.SENT_START)
                ywords.append(Vocab.SENT_END)
                x = [self.x_vocab.lookup(w) for w in xwords]
                all_x.append(x)
                all_x_lengths.append(len(x))
                y = [self.y_vocab.lookup(w) for w in ywords]
                all_y.append(y)
                all_y_lengths.append(len(y))
        all_x = np.array(all_x, dtype=object)
        all_y = np.array(all_y, dtype=object)
        lengths = np.array([max(len(x), len(y)) for x, y in zip(all_x, all_y)])
        indices = np.arange(len(lengths))
        indices = indices[np.argsort(lengths[indices], kind='mergsort')]
        return all_x[indices], all_y[indices]

    def _make_batches(self, batch_max_words, batch_max_sentences):
        batches = []
        batch_start = 0
        batch_end = 0
        batch_width = 0
        for x, y in zip(self.all_x, self.all_y):
            length_if_added = (batch_end - batch_start) + 1
            width_if_added = max(batch_width, len(x), len(y))
            words_if_added = length_if_added * width_if_added
            if (length_if_added > batch_max_sentences) or \
                    (words_if_added > batch_max_words):
                batches.append((batch_start, batch_end))
                batch_start = batch_end
                batch_width = 0
            batch_width = max(batch_width, len(x), len(y))
            batch_end += 1
        batches.append((batch_start, batch_end))
        return np.array(batches)

    def _prepare_data(self, batch_start, batch_end):
        seqs_x = self.all_x[batch_start:batch_end]
        seqs_y = self.all_y[batch_start:batch_end]
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]
        n_samples = len(seqs_y)
        maxlen_x = np.max(lengths_x)
        maxlen_y = np.max(lengths_y)
        x = np.zeros((n_samples, maxlen_x)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        y = np.zeros((n_samples, maxlen_y)).astype('int32')
        y_mask = np.zeros((n_samples, maxlen_y - 1)).astype('float32')
        for idx, (s_x, s_y) in enumerate(zip(seqs_x, seqs_y)):
            x[idx, :lengths_x[idx]] = s_x
            x_mask[idx, :lengths_x[idx]] = 1
            y[idx, :lengths_y[idx]] = s_y
            y_mask[idx, :lengths_y[idx] - 1] = 1
        return x, x_mask, y, y_mask


class XDataset:
    def __init__(self, xpath, x_vocab, max_words_per_batch=4000, max_sentences_per_batch=200):
        self.xpath = xpath
        self.x_vocab = x_vocab
        self.all_x = self._load(xpath)
        self.batches = self._make_batches(max_words_per_batch, max_sentences_per_batch)
        log.info('loaded {} examples from {}'.format(self.size(), xpath))

    def __call__(self):
        for batch_start, batch_end in self.batches:
            yield self._prepare_data(batch_start, batch_end)

    def size(self):
        return len(self.all_x)

    def _load(self, xpath):
        all_x = []
        with open(xpath, encoding='utf8') as fx:
            for xline in fx:
                xwords = xline.rstrip().split()
                xwords.append(Vocab.SENT_END)
                x = [self.x_vocab.lookup(w) for w in xwords]
                all_x.append(x)
        return np.array(all_x, dtype=object)

    def _make_batches(self, batch_max_words, batch_max_sentences):
        batches = []
        batch_start = 0
        batch_end = 0
        batch_width = 0
        for x in self.all_x:
            length_if_added = (batch_end - batch_start) + 1
            width_if_added = max(batch_width, len(x))
            words_if_added = length_if_added * width_if_added
            if (length_if_added > batch_max_sentences) or \
                    (words_if_added > batch_max_words):
                batches.append((batch_start, batch_end))
                batch_start = batch_end
                batch_width = 0
            batch_width = max(batch_width, len(x))
            batch_end += 1
        batches.append((batch_start, batch_end))
        return np.array(batches)

    def _prepare_data(self, batch_start, batch_end):
        seqs_x = self.all_x[batch_start:batch_end]
        lengths_x = [len(s) for s in seqs_x]
        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x)
        x = np.zeros((n_samples, maxlen_x)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        for idx, s_x in enumerate(seqs_x):
            x[idx, :lengths_x[idx]] = s_x
            x_mask[idx, :lengths_x[idx]] = 1
        return x, x_mask
