#! /usr/bin/env python
#
# Train a model.

import argparse
import datetime
import json
import logging
import numpy as np
import os
import re
import shutil
import socket
import time
import math

from cnn.data_utils import batch_generator, prepare_data
from cnn.model_dir_utils import sorted_model_files, find_latest_model
from cnn.config import ModelConfiguration
from cnn.cnn_mt import CNMT
from cnn.logging_utils import init_logging
from cnn.optimizers import get_optimizer
from cnn.vocab import Vocab


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='dir for saving model data')
    parser.add_argument('train_src', help='train source sentences')
    parser.add_argument('train_tgt', help='train target sentences')
    parser.add_argument('valid_src', help='validation source sentences')
    parser.add_argument('valid_tgt', help='validation target sentences')
    parser.add_argument('--config',
                        help='config json file; required for first run')
    parser.add_argument('--valid-freq', required=True, type=int,
                        help='(default: %(default)s)')
    parser.add_argument('--optimizer', default='adam',
                        help='(default: %(default)s)')
    parser.add_argument('--learning-rate', type=float,
                        help='defaults per optimizer')
    parser.add_argument('--override-learning-rate', action='store_true',
                        help='override learning rate from saved model')
    parser.add_argument('--batch-size', type=int, default=40,
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
                        help='(default: %(default)s)')
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

    init_vocab([args.train_src, args.valid_src],
               os.path.join(args.model_dir, 'x_vocab.txt'))
    init_vocab([args.train_tgt, args.valid_tgt],
               os.path.join(args.model_dir, 'y_vocab.txt'))

    test_sentences = []
    with open(args.valid_src) as f:
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

    optimizer = get_optimizer(args.optimizer)
    learning_rate = args.learning_rate or optimizer.DEFAULT_LEARNING_RATE

    train(config, optimizer, args.model_dir, args.train_src, args.train_tgt,
          args.valid_src, args.valid_tgt, args.batch_size, args.epochs,
          test_sentences, args.test_interval, args.valid_freq,
          args.keep_models, args.patience, args.max_words,
          learning_rate, max_seconds, int(args.exit_status_max_train),
          args.anneal_restarts, args.anneal_decay, args.override_learning_rate)


class TrainingState:
    def __init__(self):
        self.completed_epochs = 0
        self.epoch_examples_seen = 0
        self.epoch_cost = 0.0
        self.training_iteration = 0
        self.bad_counter = 0
        self.validation_costs = []
        self.anneal_restarts_done = 0
        self.learning_rate = None
        self.total_train_seconds = 0

    @staticmethod
    def path_for_model(model_path):
        filename = os.path.basename(model_path)
        filename = re.sub(r'\.npz$', '.json', filename)
        return os.path.join(
            os.path.dirname(model_path), 'training-state-{}'.format(filename))

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


def train(config, optimizer, model_dir, train_src, train_tgt, valid_src,
          valid_tgt, batch_size, epochs, test_sentences, test_interval,
          valid_freq, keep_models, patience, max_words, learning_rate,
          max_seconds, exit_status_max_train, anneal_restarts, anneal_decay,
          override_learning_rate):
    start = time.time()
    state = TrainingState()
    state.learning_rate = learning_rate
    log.info('hostname: %s', socket.gethostname())
    log.info('optimizer: %s', optimizer.__class__.__qualname__)
    x_vocab = Vocab(vocab_path=os.path.join(model_dir, 'x_vocab.txt'))
    y_vocab = Vocab(vocab_path=os.path.join(model_dir, 'y_vocab.txt'))
    cnmt = CNMT(x_vocab, y_vocab, config)
    model_file = find_latest_model(model_dir)
    if model_file:
        cnmt.load_params(model_file)
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
        optimizer_path = optimizer.path_for_model(model_file)
        if os.path.exists(optimizer_path):
            optimizer.load(optimizer_path)
    log.info('TrainingState: {}'.format(state.format_for_log()))

    f_train, f_compute_cost, f_sample = cnmt.build_trainer(optimizer)

    next_test_cycle = test_interval
    early_stop = False
    first = True
    train_seconds = state.total_train_seconds

    log.info('counting examples...')
    batches = batch_generator(train_src, train_tgt, x_vocab,
                              y_vocab, batch_size, max_words)
    examples_per_epoch = get_examples_per_epoch(batches)
    log.info('examples per epoch: %d', examples_per_epoch)
    log.info('starting train loop...')

    while state.completed_epochs < epochs:
        batches = batch_generator(train_src, train_tgt, x_vocab,
                                  y_vocab, batch_size, max_words)
        if first:
            if state.epoch_examples_seen % examples_per_epoch:
                log.info('skipping past first %d examples',
                         state.epoch_examples_seen)
                example_count = 0
                for x_in, _ in batches:
                    example_count += len(x_in)
                    if example_count >= state.epoch_examples_seen:
                        break
            first = False

        for x_in, y_in in batches:
            elapsed = time.time() - start
            if max_seconds and elapsed > max_seconds:
                log.info('%d seconds elapsed in train()', elapsed)
                log.info('exiting with status %d', exit_status_max_train)
                exit(exit_status_max_train)
            state.training_iteration += 1
            state.epoch_examples_seen += len(x_in)
            x, x_mask, y_lshifted, y_rshifted, y_mask = prepare_data(
                x_in, y_in)
            batch_cost = f_train(x, x_mask, y_lshifted, y_rshifted,
                                 y_mask, state.learning_rate)
            if math.isnan(batch_cost) or math.isinf(batch_cost):
                # Hope that previous model is free of NaNs so we can restart from there.
                log.warning('invalid cost: {}'.format(batch_cost))
                log.info('exiting with status %d', exit_status_max_train)
                exit(exit_status_max_train)
            state.epoch_cost += batch_cost
            next_test_cycle -= 1
            if next_test_cycle == 0:
                test(f_sample, x_vocab, y_vocab, test_sentences, max_words)
                next_test_cycle = test_interval
            if state.training_iteration % valid_freq == 0:
                log.info('BEGIN Validating')
                valid_batches = batch_generator(
                    valid_src, valid_tgt, x_vocab, y_vocab,
                    batch_size, max_words)
                valid_cost = dataset_cost(f_compute_cost, valid_batches)
                log.info('END   Validating')
                state.validation_costs.append(float(valid_cost))
                if valid_cost <= min(state.validation_costs):
                    state.bad_counter = 0
                log.info('validation cost: {:f}, min: {:f}, bad counter: {:d}, '
                         'train seconds: {:d}'.format(
                    valid_cost, min(state.validation_costs), state.bad_counter,
                    train_seconds + int(time.time() - start)))
                model_src = save_model(cnmt, optimizer, model_dir, keep_models, state,
                                       train_seconds + int(time.time() - start))
                if valid_cost <= min(state.validation_costs):
                    log.info('New min validation: %s; saving model.npz',
                             valid_cost)
                    model_dst = os.path.join(model_dir, 'model.npz')
                    shutil.copyfile(model_src, model_dst)
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
                            best_model_path = os.path.join(model_dir, 'model.npz')
                            cnmt.load_params(best_model_path)
                        else:
                            log.info('Early Stop!')
                            early_stop = True
                            break
        if early_stop:
            # Non-zero exit status to prevent dependent queue
            # jobs from executing.
            exit(1)
        state.completed_epochs += 1
        log.info('epoch %d, epoch cost %f', state.completed_epochs, state.epoch_cost)
        state.epoch_examples_seen = 0
        save_model(cnmt, optimizer, model_dir, keep_models, state,
                   train_seconds + int(time.time() - start))
        state.epoch_cost = 0
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


def test(fast_predict, x_vocab, y_vocab, test_sentences, max_words):
    log.info('BEGIN test sentences')
    for sent in test_sentences:
        log.info('%s', sent)
        sent = sent + ' ' + Vocab.SENT_END
        x = [x_vocab.lookup(w) for w in sent.split()]
        sample, cost = fast_predict(x, max_words)
        log.info('%s', y_vocab.words_for_indexes(sample))
        log.info('')
    log.info('END   test sentences')


def dataset_cost(f_compute_cost, batch_iterator):
    batch_costs = []
    for x_in, y_in in batch_iterator:
        x, x_mask, y_lshifted, y_rshifted, y_mask = prepare_data(
            x_in, y_in)
        batch_cost = f_compute_cost(x, x_mask, y_lshifted, y_rshifted,
                                    y_mask)
        batch_costs.append(batch_cost)
    return np.mean(batch_costs)


def save_model(cnmt, optimizer, model_dir, keep_models, state, train_seconds):
    model_path = get_model_path(model_dir, state.training_iteration)
    log.info('BEGIN Saving model')
    cnmt.save_params(model_path)
    state.save(state.path_for_model(model_path), train_seconds)
    optimizer.save(optimizer.path_for_model(model_path))

    # clean all but last `keep_models` models
    paths = sorted_model_files(model_dir)
    for i in range(len(paths) - keep_models):
        if os.path.exists(paths[i]):
            log.info('removing %s', paths[i])
            os.remove(paths[i])
        flag_path = paths[i] + '.success'
        if os.path.exists(flag_path):
            log.info('removing %s', flag_path)
            os.remove(flag_path)
        state_path = state.path_for_model(paths[i])
        if os.path.exists(state_path):
            log.info('removing %s', state_path)
            os.remove(state_path)
        optimizer_path = optimizer.path_for_model(paths[i])
        if os.path.exists(optimizer_path):
            log.info('removing %s', optimizer_path)
            os.remove(optimizer_path)

    # flag file so we know we didn't get killed in the middle of
    # writing a model file
    model_success_flag = model_path + '.success'
    log.info('writing flag file: %s', model_success_flag)
    # noinspection PyUnusedLocal
    with open(model_success_flag, 'w') as f:
        pass

    log.info('END Saving model')
    return model_path


def get_model_path(model_dir, iteration):
    return os.path.join(model_dir, 'model-iter-{}.npz'.format(iteration))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        exit(1)
