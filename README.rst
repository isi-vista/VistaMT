=======
VistaMT
=======

.. contents::


Introduction
============

ISI's convolutional sequence-to-sequence software for machine translation.


Requirements
============

* Python3, tensorflow, putil
* cuda, cudnn

Tested configuration
--------------------

::

  # conda tensorflow-gpu currently has trouble finding cuda ptxas so
  # we use a mixture of conda and pip, and spack to get cuda and cudnn

  $ conda create -p ./venv python=3.6 pip
  $ conda activate ./venv
  $ pip install -r requirements.txt
  $ spack load cuda@10.1.243
  $ spack load cudnn@7.6.5.32-10.2-linux-x64

  $ nvidia-smi | grep Driver
  | NVIDIA-SMI 440.44       Driver Version: 440.44       CUDA Version: 10.2     |

If you install cuda/cudnn manually, you will need something like this at runtime:

-  CUDA_HOME=/path/to/cuda-10.1
-  CUDNN_HOME=/path/to/cudnn-7.6
-  export PATH=$CUDA_HOME/bin:$PATH
-  export CPATH="$CUDNN_HOME/cuda/include:$CUDA_HOME/include:$CPATH"
-  export LD_LIBRARY_PATH="$CUDNN_HOME/cuda/lib64/:$LD_LIBRARY_PATH"
-  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
-  export LIBRARY_PATH=$LD_LIBRARY_PATH


Training
========

Usage
-----

::

  [~/VistaMT]
  $ export PYTHONPATH=`pwd`
  $ tools/train.py -h
  usage: train.py [-h] [--valid-ref VALID_REF] [--lc-bleu] [--stop-on-cost]
                  [--config CONFIG] --valid-freq VALID_FREQ
                  [--learning-rate LEARNING_RATE] [--override-learning-rate]
                  --batch-max-words BATCH_MAX_WORDS --batch-max-sentences
                  BATCH_MAX_SENTENCES [--epochs EPOCHS]
                  [--test-interval TEST_INTERVAL] [--test-count TEST_COUNT]
                  [--keep-models KEEP_MODELS] [--patience PATIENCE]
                  [--anneal-restarts ANNEAL_RESTARTS]
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
    --valid-ref VALID_REF
                          validation ref sentences for greedy BLEU
    --lc-bleu             lowercase BLEU
    --stop-on-cost        use cost for stopping criteria
    --config CONFIG       config json file; required for first run
    --valid-freq VALID_FREQ
                          (default: None)
    --learning-rate LEARNING_RATE
                          (default: 0.0002)
    --override-learning-rate
                          override learning rate from saved model
    --batch-max-words BATCH_MAX_WORDS
                          (default: 4000)
    --batch-max-sentences BATCH_MAX_SENTENCES
                          (default: 200)
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
                          discard long sentences (default: 50)
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
  ...
  models/cnn01/model-iter-165000.data-00000-of-00002
  models/cnn01/model-iter-165000.data-00001-of-00002
  models/cnn01/model-iter-165000.index
  models/cnn01/model-iter-165000.success
  models/cnn01/model-iter-165000.training-state.json
  models/cnn01/model.data-00000-of-00002
  models/cnn01/model.data-00001-of-00002
  models/cnn01/model.index
  models/cnn01/model.success
  models/cnn01/model.training-state.json
  train.log

Models are written to disk after every validation run.  The models are
named with the iteration number.  Only the last ``keep_models`` models
are kept since the sizes can be large.  A ``.success`` file is written
after the model itself is written so the user can be sure training was
not stopped in the middle of writing a model file.  A training state
file is also written with each model so that training can be
restarted.

The iteration with the best performance is kept with the ``model.`
prefix`.  If ``--valid-ref`` is given performance is measured as the
max greedy BLEU score.  Otherwise the minimum validation cost is used.

When a training run is restarted, it uses the latest iteration files
in the MODEL_DIR as a starting point.  The MODEL_DIR/config.json file
is a copy of the config file used when training began.

Parameters like ``patience`` or ``epochs`` can be changed.  After a
typical training run completes, you may indeed need to increase these
otherwise training may immediately stop.

Learning rate can be changed on restart by passing both
``--learning-rate`` and ``override-learning-rate``.  The latter is a
boolean flag that forces the provided learning rate to take effect.
The default behavior is to take learning rate from the saved model
state, since the learning rate is adjusted automatically during a
typical training run.

Configuration
-------------

The structural configuration of the model is specified in a JSON file
which looks like this:

::

  [~/VistaMT]
  $ cat sample-config.json
  {
    "emb_dim": 512,
    "out_emb_dim": 512,
    "dropout_rate": 0.3,
    "encoder_arch": [[15,3,512]],
    "decoder_arch": [[10,3,512]],
    "num_positions": 256,
    "num_attn_heads": 4
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

``num_positions`` is the maximum number of positions available for the
position embeddings.

``num_attn_heads`` is the number of heads used for multi-head
attention.


Example
-------

::

  [~/VistaMT]
  $ export PYTHONPATH=`pwd`
  $ python tools/train.py model_dir \
  ro-en/train.ro ro-en/train.en ro-en/valid.ro ro-en/valid.en \
  --valid-freq 2000 --batch-max-words 6000 --batch-max-sentences 200 \
  --test-interval 50000 --config sample-config.json

Training procedure
------------------

Training continues until ``epochs`` epochs are completed or an early
stop is detected.

During training, a ``bad_counter`` keeps track of the number of times
the validation cost exceeds the minimum validation cost, or the number
of times greedy BLEU is less than the best greedy BLEU, if
``--valid-ref`` is passed.  If this counter exceeds the ``patience``
threshold, the parameters are reset to the best ones found so far and
the learning rate is reduced (by ``anneal-decay``).  After this
restarting happens ``anneal-restarts`` times, if ``patience`` is
exceeded again, training stops.

Batching is done by grouping training examples by their length.  All
batches are read into memory, then they are shuffled randomly on every
epoch.  The batch size is variable, depending on the
``--batch-max-words`` and ``--batch-max-sentences`` parameters.


Prediction
==========

Usage
-----

::

  [~/VistaMT]
  $ export PYTHONPATH=`pwd`
  $ tools/predict.py -h
  usage: predict.py [-h] [--beam-size BEAM_SIZE] [--max-words MAX_WORDS]
                    [--model-filename MODEL_FILENAME] [--log-level LOG_LEVEL]
                    [--log-file LOG_FILE] [--batch-greedy]
                    [--batch-size BATCH_SIZE]
                    [--batch-max-words BATCH_MAX_WORDS] [--nbest]
                    model_dir src tgt

  positional arguments:
    model_dir
    src
    tgt

  optional arguments:
    -h, --help            show this help message and exit
    --beam-size BEAM_SIZE
                          (default: 10)
    --max-words MAX_WORDS
                          (default: 80)
    --model-filename MODEL_FILENAME
                          use specific model instead of latest iter
    --log-level LOG_LEVEL
                          (default: INFO
    --log-file LOG_FILE   (default: predict-{tgt}.log)
    --batch-greedy        greedy decode on batches of sentences at once
    --batch-size BATCH_SIZE
                          batch size for --batch-greedy (default: 80)
    --batch-max-words BATCH_MAX_WORDS
                          (default: 4000)
    --nbest               write nbest list; n = beam-size

Prediction uses the latest iteration model file by default.  You can
use the model with the best validation score by passing
``--model-filename model.npz``.


Example
-------

::

  [~/VistaMT]
  $ python tools/predict.py model_dir \
  wmt17-preprocessed/newstest2017.bpe.ru newstest2017.bpe.en.predicted.out


Acknowledgments
==========

The research is based upon work supported by the Office of the Director of
National Intelligence (ODNI), Intelligence Advanced Research Projects
Activity (IARPA), via AFRL Contract #FA8650-17-C-9116.
The views and conclusions contained herein are those of the authors and
should not be interpreted as necessarily representing the official policies or
endorsements, either expressed or implied, of the ODNI, IARPA, or the
U.S. Government. The U.S. Government is authorized to reproduce and
distribute reprints for Governmental purposes notwithstanding any
copyright annotation thereon.
