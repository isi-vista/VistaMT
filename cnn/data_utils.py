import argparse
import logging

import numpy as np
from numpy.random.mtrand import RandomState

from cnn.logging_utils import init_logging
from cnn.vocab import Vocab


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Dataset:
    def __init__(self, xpath, ypath, x_vocab, y_vocab, max_words):
        self.xpath = xpath
        self.ypath = ypath
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.max_words = max_words
        self.all_x, self.all_y = self._load(xpath, ypath)
        log.info('loaded {} examples from {}, {}'.format(self.size(), xpath, ypath))

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
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        sentence_indices = np.argsort(all_x_lengths, kind='mergsort')
        return all_x[sentence_indices], all_y[sentence_indices]

    # iterates over tuples of: x, x_mask, y_lshifted, y_rshifted, y_mask
    def iterator(self, batch_max_words, batch_max_sentences, random_state):
        bg = self._batch_generator(batch_max_words, batch_max_sentences)
        all_prepared_batches = [self._prepare_data(x, y) for x, y in bg]
        if random_state:
            random_state.shuffle(all_prepared_batches)
        log.info('prepared {} {}batches for {}, {}'.format(
            len(all_prepared_batches),
            'shuffled ' if random_state else '',
            self.xpath, self.ypath))
        return iter(all_prepared_batches)

    # iterates over binarized batched sentences: seqs_x, seqs_y
    def _batch_generator(self, batch_max_words, batch_max_sentences):
        seqs_x = []
        seqs_y = []
        x_token_count = 0
        y_token_count = 0
        for x, y in zip(self.all_x, self.all_y):
            if ((x_token_count + len(x) >= batch_max_words)
                    or (y_token_count + len(y) >= batch_max_words)
                    or (len(seqs_x) >= batch_max_sentences)):
                yield seqs_x, seqs_y
                seqs_x.clear()
                seqs_y.clear()
                x_token_count = 0
                y_token_count = 0
            seqs_x.append(x)
            seqs_y.append(y)
            x_token_count += len(x)
            y_token_count += len(y)
        if seqs_x:
            yield seqs_x, seqs_y

    @staticmethod
    def _prepare_data(seqs_x, seqs_y):
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
            if len(xwords) > max_words:
                continue
            xwords.append(Vocab.SENT_END)
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


# just for testing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('x_vocab')
    parser.add_argument('y_vocab')
    parser.add_argument('x')
    parser.add_argument('y')
    parser.add_argument('--batch-max-words', type=int, required=True)
    parser.add_argument('--batch-max-sentences', type=int, required=True)
    parser.add_argument('--max-words', type=int, default=80,
                        help='sentences longer than this are discarded')
    parser.add_argument('--shuffle-batches', dest='shuffle_batches', action='store_true')
    parser.add_argument('--no-shuffle-batches', dest='shuffle_batches', action='store_false')
    parser.set_defaults(shuffle_batches=True)
    parser.add_argument('--log-level', default='INFO',
                        help='(default: %(default)s)')
    args = parser.parse_args()

    init_logging(None, args.log_level)
    log.info('command line args: {}'.format(args))

    x_vocab = Vocab(vocab_path=args.x_vocab)
    y_vocab = Vocab(vocab_path=args.y_vocab)

    dataset = Dataset(args.x, args.y, x_vocab, y_vocab, args.max_words)
    print(dataset.size())
    random_state = RandomState() if args.shuffle_batches else None
    batches = dataset.iterator(args.batch_max_sentences, args.max_words,
                               random_state)
    for b in batches:
        x, _, y_lshifted, _, _ = b
        print(x.shape, y_lshifted.shape)


if __name__ == '__main__':
    main()
