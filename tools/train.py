#! /usr/bin/env python
#
# Train a model.

import argparse
import datetime
import json
import logging
import subprocess
import tempfile

import numpy as np
import os
import psutil
import re
import shutil
import socket
import time

from numpy.random.mtrand import RandomState

from cnn import compat
from cnn.cnn_mt import ConvolutionalMT, DEFAULT_LEARNING_RATE
from cnn.data_utils import XYDataset, XDataset
from cnn.model_dir_utils import sorted_model_files, find_latest_model, MODEL_PREFIX, \
    model_iter_from_path, copy_checkpoint, path_to_checkpoint, checkpoint_to_paths, \
    TRAINING_STATE_SUFFIX, SUCCESS_SUFFIX
from cnn.config import ModelConfiguration
from cnn.logging_utils import init_logging
from cnn.vocab import Vocab
from tools.predict import batch_predict

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def used_memory_in_gigabytes():
    process = psutil.Process(os.getpid())
    nbytes = process.memory_info().rss
    return nbytes / (1024 * 1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='dir for saving model data')
    parser.add_argument('train_src', help='train source sentences')
    parser.add_argument('train_tgt', help='train target sentences')
    parser.add_argument('valid_src', help='validation source sentences')
    parser.add_argument('valid_tgt', help='validation target sentences')
    parser.add_argument('--valid-ref',
                        help='validation ref sentences for greedy BLEU')
    parser.add_argument('--lc-bleu', default=False, action='store_true',
                        help='lowercase BLEU')
    parser.add_argument('--stop-on-cost', default=False, action='store_true',
                        help='use cost for stopping criteria')
    parser.add_argument('--config',
                        help='config json file; required for first run')
    parser.add_argument('--valid-freq', required=True, type=int,
                        help='(default: %(default)s)')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='(default: %(default)s)')
    parser.add_argument('--override-learning-rate', action='store_true',
                        help='override learning rate from saved model')
    parser.add_argument('--batch-max-words', type=int, required=True, default=4000,
                        help='(default: %(default)s)')
    parser.add_argument('--batch-max-sentences', type=int, required=True, default=200,
                        help='(default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='(default: %(default)s)')
    parser.add_argument('--test-interval', type=int, default=500,
                        help='(default: %(default)s)')
    parser.add_argument('--test-count', type=int, default=10,
                        help='(default: %(default)s)')
    parser.add_argument('--keep-models', type=int, default=3,
                        help='(default: %(default)s)')
    parser.add_argument('--patience', type=int, default=10,
                        help='(default: %(default)s)')
    parser.add_argument('--anneal-restarts', type=int, default=2,
                        help='(default: %(default)s)')
    parser.add_argument('--anneal-decay', type=float, default=0.5,
                        help='(default: %(default)s)')
    parser.add_argument('--max-words', type=int, default=50,
                        help='discard long sentences (default: %(default)s)')
    parser.add_argument('--log-level', default='INFO',
                        help='(default: %(default)s)')
    parser.add_argument('--log-file',
                        help='(default: model_dir/train.log)')
    parser.add_argument('--max-train-duration',
                        help='days:hrs:mins:secs; exit after duration elapses')
    parser.add_argument('--exit-status-max-train', default=99,
                        help='(default: %(default)s)')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not args.log_file:
        args.log_file = os.path.join(args.model_dir, 'train.log')
    init_logging(args.log_file, args.log_level)
    log.info('command line args: {}'.format(args))

    init_vocab([args.train_src],
               os.path.join(args.model_dir, 'x_vocab.txt'))
    init_vocab([args.train_tgt],
               os.path.join(args.model_dir, 'y_vocab.txt'))

    test_sentences = []
    with open(args.valid_src, encoding='utf8') as f:
        for i, line in enumerate(f):
            if i == args.test_count:
                break
            test_sentences.append(line.rstrip())

    config_path = os.path.join(args.model_dir, 'config.json')
    if os.path.exists(config_path):
        if args.config:
            log.warning('{} exists; ignoring --config'.format(config_path))
        config = ModelConfiguration(config_path)
    elif not args.config:
        raise ValueError('--config is required when training new model')
    else:
        config = ModelConfiguration(args.config)
        shutil.copyfile(args.config, config_path)

    # need space for all words plus start and end tokens
    if args.max_words > config.num_positions - 2:
        raise ValueError('args.max-words must be <= config.num_positions - 2')

    max_seconds = 0
    if args.max_train_duration:
        # days:hrs:mins:secs with all optional except secs
        m = re.match(r'(?:(?:(?:(\d+):)?(\d+):)?(\d+):)?(\d+)$',
                     args.max_train_duration)
        if not m:
            raise ValueError('invalid time duration: {}'.format(args.max_train_duration))

        def parse_int(s):
            if s is None:
                return 0
            return int(s)

        max_seconds = datetime.timedelta(
            days=parse_int(m.groups()[0]),
            hours=parse_int(m.groups()[1]),
            minutes=parse_int(m.groups()[2]),
            seconds=parse_int(m.groups()[3])).total_seconds()
        max_seconds = int(max_seconds)
        logging.info('Will exit training after {} seconds'.format(max_seconds))

    learning_rate = args.learning_rate
    stop_on_cost = args.stop_on_cost or args.valid_ref is None

    train(config, args.model_dir, args.train_src, args.train_tgt,
          args.valid_src, args.valid_tgt, args.batch_max_words, args.batch_max_sentences,
          args.epochs, test_sentences, args.test_interval, args.valid_freq,
          args.keep_models, args.patience, args.max_words,
          learning_rate, max_seconds, int(args.exit_status_max_train),
          args.anneal_restarts, args.anneal_decay, args.override_learning_rate,
          args.valid_ref, args.lc_bleu, stop_on_cost)


class TrainingState:
    def __init__(self):
        self.completed_epochs = 0
        self.training_iteration = 0
        self.bad_counter = 0
        self.validation_costs = []
        self.validation_bleus = []
        self.anneal_restarts_done = 0
        self.learning_rate = None
        self.total_train_seconds = 0

    @staticmethod
    def path_for_model(model_path):
        model_path = path_to_checkpoint(model_path)
        return model_path + TRAINING_STATE_SUFFIX

    def to_json(self, train_seconds=None, **kwargs):
        d = dict(vars(self))
        if train_seconds:
            d['total_train_seconds'] = train_seconds
        return json.dumps(d, **kwargs)

    def save(self, path, train_seconds):
        log.info('Saving training state to {}'.format(path))
        with open(path, 'w', encoding='utf8') as f:
            print(self.to_json(train_seconds=train_seconds, indent=2), file=f)

    def load(self, path):
        log.info('Loading training state from {}'.format(path))
        with open(path, encoding='utf8') as f:
            d = json.load(f)
        for key in d:
            setattr(self, key, d[key])

    def format_for_log(self):
        s = self.to_json()
        d = json.loads(s)
        if 'validation_costs' in d:
            costs = d['validation_costs']
            limit = 5
            if len(costs) > limit:
                d['validation_costs'] = ['...']
                d['validation_costs'].extend(costs[-limit:])
        return json.dumps(d, indent=2)


def compute_greedy_bleu(cnn_mt, valid_dataset, valid_ref, lc_bleu, max_words):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8') as tf:
        batch_predict(cnn_mt, valid_dataset, tf.name, max_words, beam_size=1)
        cmd = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'scripts', 'score.sh'),
               valid_ref, tf.name]
        if lc_bleu:
            cmd.insert(1, '-lc')
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode:
            log.error(str(p.stderr, encoding='utf8'))
            p.check_returncode()
        output_lines = str(p.stdout, encoding='utf8').split('\n')
        # BLEU = 0.00, 2.8/0.0/0.0/0.0 (BP=1.000, ratio=1.323, hyp_len=500, ref_len=378)
        bleu_line = ''
        for line in output_lines:
            if line.startswith('BLEU'):
                bleu_line = line.rstrip()
                if lc_bleu:
                    bleu_line = bleu_line.replace('BLEU', 'BLEU-lc')
                break
        m = re.search(r'= (\d+\.\d+),', bleu_line)
        bleu = float(m.group(1))
        return bleu, bleu_line


def train(config, model_dir, train_src, train_tgt, valid_src,
          valid_tgt, batch_max_words, batch_max_sentences, epochs, test_sentences,
          test_interval, valid_freq, keep_models, patience, max_words, learning_rate,
          max_seconds, exit_status_max_train, anneal_restarts, anneal_decay,
          override_learning_rate, valid_ref, lc_bleu, stop_on_cost):
    start = time.time()
    state = TrainingState()
    state.learning_rate = learning_rate
    log.info('hostname: %s', socket.gethostname())
    x_vocab = Vocab(vocab_path=os.path.join(model_dir, 'x_vocab.txt'))
    y_vocab = Vocab(vocab_path=os.path.join(model_dir, 'y_vocab.txt'))

    cnn_mt = ConvolutionalMT(config, x_vocab, y_vocab)
    model_file = find_latest_model(model_dir)
    if model_file:
        compat.load_params(cnn_mt, model_file)
        state_path = state.path_for_model(model_file)
        if os.path.exists(state_path):
            state.load(state_path)
            if state.learning_rate != learning_rate:
                if override_learning_rate:
                    log.info('overriding saved learning rate {} to {}'.format(
                        state.learning_rate, learning_rate))
                    state.learning_rate = learning_rate
                else:
                    log.warning('using saved learning rate {}'.format(
                        state.learning_rate))
        else:
            log.warning('no training state file found for model!')
            state.training_iteration = model_iter_from_path(model_file)
    log.info('TrainingState: {}'.format(state.format_for_log()))
    log.info('using {} for stopping criteria'.format('cost' if stop_on_cost else 'bleu'))

    next_test_cycle = test_interval
    early_stop = False
    train_seconds = state.total_train_seconds

    # Get a different random state to avoid seeing the same shuffled batches
    # on restart.  We want to see different data, especially for large datasets.
    random_state = RandomState()

    log.info('preparing training batches...')
    train_dataset = XYDataset(train_src, train_tgt, x_vocab, y_vocab,
                              max_words_per_sentence=max_words,
                              max_words_per_batch=batch_max_words,
                              max_sentences_per_batch=batch_max_sentences,
                              random_state=random_state)
    log.info('preparing validation batches...')
    valid_xy_dataset = XYDataset(valid_src, valid_tgt, x_vocab, y_vocab,
                                 max_words_per_sentence=max_words,
                                 max_words_per_batch=batch_max_words,
                                 max_sentences_per_batch=batch_max_sentences,
                                 random_state=None)
    valid_x_dataset = XDataset(valid_src, x_vocab, config.num_positions,
                               max_words_per_batch=batch_max_words,
                               max_sentences_per_batch=batch_max_sentences)

    log.info('starting train loop...')
    log.info('process memory at start of train loop: {:.2f} GB'.format(
        used_memory_in_gigabytes()))

    while state.completed_epochs < epochs:
        epoch_cost = 0
        for batch in train_dataset():
            x, x_mask, y, y_mask = batch
            elapsed = time.time() - start
            if max_seconds and elapsed > max_seconds:
                log.info('%d seconds elapsed in train()', elapsed)
                log.info('exiting with status %d', exit_status_max_train)
                exit(exit_status_max_train)
            state.training_iteration += 1

            cnn_mt.set_learning_rate(state.learning_rate)
            batch_cost = cnn_mt.train(x, x_mask, y, y_mask)

            epoch_cost += batch_cost
            next_test_cycle -= 1
            if next_test_cycle == 0:
                test(cnn_mt, x_vocab, y_vocab, test_sentences, max_words)
                next_test_cycle = test_interval
            if state.training_iteration % valid_freq == 0:
                log.info('BEGIN Validating')
                valid_cost = dataset_cost(cnn_mt, valid_xy_dataset)
                state.validation_costs.append(float(valid_cost))
                new_best = False
                bleu, bleu_s, max_bleu_s = -1.0, '?????', '?????'
                if valid_ref:
                    bleu, bleu_line = compute_greedy_bleu(cnn_mt, valid_x_dataset, valid_ref,
                                                          lc_bleu, max_words)
                    log.info(bleu_line)
                    state.validation_bleus.append(bleu)
                    bleu_s = '{:05.2f}'.format(bleu)
                    max_bleu_s = '{:05.2f}'.format(max(state.validation_bleus))
                if stop_on_cost:
                    if valid_cost <= min(state.validation_costs):
                        state.bad_counter = 0
                        new_best = True
                else:
                    if bleu >= max(state.validation_bleus):
                        state.bad_counter = 0
                        new_best = True
                log.info('END   Validating')
                ts = train_seconds + int(time.time() - start)
                log.info('bleu{} {:5s} max {:5s} cost {:f} min {:f} bad_counter {:d} lr {:f} '
                         'iter {:d} completed_epochs: {:d} train_secs {:d}'.format(
                          '-lc' if lc_bleu else '', bleu_s, max_bleu_s, valid_cost,
                          min(state.validation_costs), state.bad_counter, state.learning_rate,
                          state.training_iteration, state.completed_epochs, ts))
                model_src = save_model(cnn_mt, model_dir, keep_models, state,
                                       train_seconds + int(time.time() - start))
                if new_best:
                    log.info('New best model; saving model')
                    model_dst = os.path.join(model_dir, 'model')
                    copy_checkpoint(model_src, model_dst)
                else:
                    state.bad_counter += 1
                    if state.bad_counter > patience:
                        if state.anneal_restarts_done < anneal_restarts:
                            log.info('No progress on the validation set, annealing learning '
                                     'rate and resuming from best params.')
                            state.learning_rate *= anneal_decay
                            log.info('new learning rate: {:f}'.format(state.learning_rate))
                            state.anneal_restarts_done += 1
                            state.bad_counter = 0
                            best_model_path = os.path.join(model_dir, 'model')
                            compat.load_params(cnn_mt, best_model_path)
                        else:
                            log.info('Early Stop!')
                            early_stop = True
                            break
        if early_stop:
            # Non-zero exit status to prevent dependent queue
            # jobs from executing.
            exit(1)
        state.completed_epochs += 1
        log.info('epoch %d, epoch cost %f', state.completed_epochs, epoch_cost)
        log.info('process memory at end of epoch: {:.2f} GB'.format(
            used_memory_in_gigabytes()))
    log.info('process memory at end of training: {:.2f} GB'.format(
        used_memory_in_gigabytes()))
    log.info('training ends')


def init_vocab(src_paths, dst_path):
    if not os.path.exists(dst_path):
        logging.info('BEGIN Creating vocab: %s', dst_path)
        Vocab(src_paths).write(dst_path)
        logging.info('END   Creating vocab')
    else:
        logging.info('using existing vocab: %s', dst_path)


def get_examples_per_epoch(batches):
    return sum([len(x) for x, _ in batches])


def test(cnn_mt, x_vocab, y_vocab, test_sentences, max_words):
    log.info('BEGIN test sentences')
    for sent in test_sentences:
        log.info('%s', sent)
        sent = sent + ' ' + Vocab.SENT_END
        x = [x_vocab.lookup(w) for w in sent.split()]
        batch_x = np.array([x], dtype=np.int32)
        batch_x_mask = np.ones_like(batch_x, dtype=np.float32)
        sample = cnn_mt.predict(batch_x, batch_x_mask, max_words, 1)[0]
        log.info('%s', y_vocab.words_for_indexes(sample))
        log.info('')
    log.info('END   test sentences')


def dataset_cost(cnn_mt, dataset):
    batch_costs = []
    for batch in dataset():
        x, x_mask, y, y_mask = batch
        batch_cost = cnn_mt.get_cost(x, x_mask, y, y_mask)
        batch_costs.append(batch_cost)
    return np.mean(batch_costs)


def save_model(cnn_mt, model_dir, keep_models, state, train_seconds):
    model_path = get_model_path(model_dir, state.training_iteration)
    log.info('BEGIN Saving model')
    compat.save_params(cnn_mt, model_path)
    state.save(state.path_for_model(model_path), train_seconds)

    # flag file so we know we didn't get killed in the middle of
    # writing a model
    model_success_flag = model_path + SUCCESS_SUFFIX
    log.info('writing flag file: %s', model_success_flag)
    # noinspection PyUnusedLocal
    with open(model_success_flag, 'w') as f:
        pass

    # tensorflow Saver can manage delete of old checkpoints with max_to_keep,
    # but it seems to have issues on restarts so we do it ourselves.  We also
    # have to clean up associated state files.

    # clean all but last `keep_models` models
    paths = sorted_model_files(model_dir)
    for i in range(len(paths) - keep_models):
        prefix = paths[i]
        for path in checkpoint_to_paths(prefix):
            if os.path.exists(path):
                log.info('removing %s', path)
                os.remove(path)

    log.info('END Saving model')
    return model_path


def get_model_path(model_dir, iteration):
    return os.path.join(model_dir, '{}{}'.format(MODEL_PREFIX, iteration))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        exit(1)
