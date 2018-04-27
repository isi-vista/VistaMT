=======
VistaMT
=======

.. contents::


Introduction
============

ISI's convolutional sequence-to-sequence software for machine translation.


Requirements
============

* Python3, numpy, theano

Tested configuration
--------------------

::

  $ python -V
  Python 3.5.4 :: Anaconda custom (64-bit)

  $ conda list | egrep -i '(numpy|theano|pygpu|libgpuarray)'
  libgpuarray               0.7.5                h14c3975_0
  numpy                     1.14.1                    <pip>
  numpy                     1.12.1           py35hca0bb5e_1
  pygpu                     0.7.5            py35h14c3975_0
  Theano                    1.0.1                     <pip>

  $ nvidia-smi | grep Driver
  | NVIDIA-SMI 384.90                 Driver Version: 384.90                    |

Also:

* cuda-8.0

* cudnn-8.0-linux-x64-v5.1

Example CUDA/cuDNN setup::

  CUDA_HOME=/usr/local/cuda-8.0
  CUDNN_HOME=/nfs/isicvlnas01/share/cudnn-8.0-linux-x64-v5.1
  export PATH=$CUDA_HOME/bin:$PATH
  export CPATH="$CUDNN_HOME/cuda/include:$CUDA_HOME/include:$CPATH"
  export LD_LIBRARY_PATH="$CUDNN_HOME/cuda/lib64/:$LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"


Training
========

Usage
-----

::

  [~/VistaMT]
  $ export PYTHONPATH=`pwd`
  $ tools/train.py -h
  usage: train.py [-h] [--config CONFIG] --valid-freq VALID_FREQ
                  [--optimizer OPTIMIZER] [--learning-rate LEARNING_RATE]
                  [--override-learning-rate] [--batch-size BATCH_SIZE]
                  [--epochs EPOCHS] [--test-interval TEST_INTERVAL]
                  [--test-count TEST_COUNT] [--keep-models KEEP_MODELS]
                  [--patience PATIENCE] [--anneal-restarts ANNEAL_RESTARTS]
                  [--anneal-decay ANNEAL_DECAY] [--max-words MAX_WORDS]
                  [--log-level LOG_LEVEL] [--log-file LOG_FILE]
                  [--max-train-duration MAX_TRAIN_DURATION]
                  [--exit-status-max-train EXIT_STATUS_MAX_TRAIN]
                  model_dir train_src train_tgt valid_src valid_tgt

  positional arguments:
    model_dir             dir for saving model data
    train_src             train source sentences
    train_tgt             train target sentences
    valid_src             validation source sentences
    valid_tgt             validation target sentences

  optional arguments:
    -h, --help            show this help message and exit
    --config CONFIG       config json file; required for first run
    --valid-freq VALID_FREQ
                          (default: None)
    --optimizer OPTIMIZER
                          (default: adam)
    --learning-rate LEARNING_RATE
                          defaults per optimizer
    --override-learning-rate
                          override learning rate from saved model
    --batch-size BATCH_SIZE
                          (default: 40)
    --epochs EPOCHS       (default: 100)
    --test-interval TEST_INTERVAL
                          (default: 500)
    --test-count TEST_COUNT
                          (default: 10)
    --keep-models KEEP_MODELS
                          (default: 3)
    --patience PATIENCE   (default: 10)
    --anneal-restarts ANNEAL_RESTARTS
                          (default: 2)
    --anneal-decay ANNEAL_DECAY
                          (default: 0.5)
    --max-words MAX_WORDS
                          (default: 50)
    --log-level LOG_LEVEL
                          (default: INFO)
    --log-file LOG_FILE   (default: model_dir/train.log)
    --max-train-duration MAX_TRAIN_DURATION
                          days:hrs:mins:secs; exit after duration elapses
    --exit-status-max-train EXIT_STATUS_MAX_TRAIN
                          (default: 99)

After a typical run, the MODEL_DIR will looks like this::

  $ ls -1rt model_dir
  x_vocab.txt
  y_vocab.txt
  config.json
  model-iter-60000.npz
  training-state-model-iter-60000.json
  adam-optimizer-model-iter-60000.npz
  model-iter-60000.npz.success
  model-iter-65000.npz
  training-state-model-iter-65000.json
  adam-optimizer-model-iter-65000.npz
  model-iter-65000.npz.success
  model-iter-70000.npz
  training-state-model-iter-70000.json
  adam-optimizer-model-iter-70000.npz
  model-iter-70000.npz.success
  train.log
  model.npz

Models are written to disk after every validation run and after every
epoch completes.  The models are named with the iteration number.
Only the last ``keep_models`` models are kept since the sizes can be
large.  A ``.success`` file is written after the model itself is
written so the user can be sure training was not stopped in the middle
of writing a model file.  For the Adam optimizer, an additional state
file is saved which allows subsequent training runs to continue with
the proper state of Adam parameters.  A training state file is also
written with each model so that training can be restarted.

The iteration with the lowest validation score so far is kept as
``model.npz``.

When a training run is restarted, it uses the latest iteration files
in the MODEL_DIR as a starting point.  The MODEL_DIR/config.json file
is a copy of the config file used when training begain.

Parameters like ``patience`` or ``epochs`` can be changed.  After a
typical training run completes, you may indeed need to increase these
otherwise training may immediately stop.

Be careful changing the optimzier (e.g. Adam keeps state) or the batch
size since iteration numbers will change meaning.

Learning rate can be changed on restart by passing both
``--learning-rate`` and ``override-learning-rate``.  The latter is a
boolean flag that forces the provided learning rate to take effet.
The default behavior is to take learning rate from the saved model
state, since the learning rate is adjusted automatically during a
typical training run.

Configuration
-------------

The structural configuration of the model is specified in a JSON file
which looks like this:

::

  $ cat sample-config.json
  {
    "emb_dim": 512,
    "out_emb_dim": 512,
    "dropout_rate": 0.2,
    "encoder_arch": [[15,3,512]],
    "decoder_arch": [[10,3,512]]
  }

This holds the structural configuration of the static graph; these
parameters cannot be changed after training has started.  Note that
dropout rate is part of this static graph.

``emb_dim`` is the dimension of the input embedding.

``out_emb_dim`` is the dimension of the output embedding.

``dropout_rate`` is a float greater than 0 and less than 1.

``encoder_arch`` and ``decoder_arch`` are lists of triples of the form
``[depth, width, dimension]``.

For example, ``[[5, 3, 512], [3, 5, 768], [2, 3, 1024]]`` specifies 5
layers of 3-wide convolutions with 512 dimension embeddings, followed
by 3 layers of 5-wide convolutions with 768 dimension embeddings,
finally followed by 2 layers of 3-wide convolutions with 1024
dimension embeddings.

Example
-------

::

  $ export PYTHONPATH=`pwd`
  $ THEANO_FLAGS=device=cuda,floatX=float32 python train.py model_dir \
  ro-en.en.bpe ro-en.ro.bpe ro-en-data/newsdev_head.ro ro-en-data/newsdev_head.ro \
  --valid-freq 2000 --batch-size 80 --test-interval 50000 --config sample-config.json

Training procedure
------------------

Training continues until ``epochs`` epochs are completed or an early
stop is detected.

During training, a ``bad_counter`` keeps track of the number of times
the validation cost exceeds the minimum validation cost so far.  If
this counter exceeds the ``patience`` threshold, the parameters are
reset to the best ones found so far (the ones that produced the
minimum validation cost) and the learning rate is reduced (by
``anneal-decay``).  After this restarting happens ``anneal-restarts``
times, if ``patience`` is exceeded again, training stops.


Prediction
==========

Usage
-----

::

  [~/VistaMT]
  $ export PYTHONPATH=`pwd`
  $ tools/predict.py -h
  usage: predict.py [-h] [--beam-width BEAM_WIDTH] [--max-words MAX_WORDS]
                    [--model-filename MODEL_FILENAME] [--log-level LOG_LEVEL]
                    [--log-file LOG_FILE] [--batch-greedy]
                    [--batch-size BATCH_SIZE]
                    model_dir src tgt

  positional arguments:
    model_dir
    src
    tgt

  optional arguments:
    -h, --help            show this help message and exit
    --beam-width BEAM_WIDTH
                          (default: 10)
    --max-words MAX_WORDS
                          (default: 50)
    --model-filename MODEL_FILENAME
                          use specific model instead of latest iter
    --log-level LOG_LEVEL
                          (default: INFO
    --log-file LOG_FILE   (default: model_dir/predict.log)
    --batch-greedy        greedy decode on batches of sentences at once
    --batch-size BATCH_SIZE
                          batch size for --batch-greedy (default: 80)

Prediction uses the latest iteration model file by default.  You can
use the model with the best validation score by passing
``--model-filename model.npz``.

The ``--batch-greedy`` option decodes at much faster speed but with
reduced accuracy.  It performs a 1-best search instead of a beam
search and is most useful for tasks such as back-translation.


Example
-------

::

  [~/VistaMT]
  $ THEANO_FLAGS=device=cuda,floatX=float32 python predict.py \
  model_dir wmt17-preprocessed/newstest2017.bpe.ru newstest2017.bpe.en.predicted.out
