import glob
import os
import re
import shutil


MODEL_PREFIX = 'model-iter-'
TRAINING_STATE_SUFFIX = '.training-state.json'
SUCCESS_SUFFIX = '.success'


def find_latest_model(model_dir):
    model_files = sorted_model_files(model_dir)
    if model_files:
        return model_files[-1]
    return None


def model_iter_from_path(path):
    m = re.match(r'{}(\d+)'.format(MODEL_PREFIX), os.path.basename(path))
    model_iter = None
    if m:
        model_iter = int(m.group(1))
    return model_iter


def sorted_model_files(model_dir):
    names = glob.glob(os.path.join(model_dir, MODEL_PREFIX + '*' + SUCCESS_SUFFIX))
    names = [re.sub(SUCCESS_SUFFIX + '$', '', n) for n in names]
    items = []
    for name in names:
        model_iter = model_iter_from_path(name)
        if model_iter:
            items.append((model_iter, name))
    items.sort()
    return [x[1] for x in items]


def path_to_checkpoint(path):
    return re.sub(r'\.(index|meta|data[^.]*)$', '', path)


def checkpoint_to_paths(checkpoint_prefix):
    paths = []
    for suffix in ['.index', '.meta', '.data-*', '.training-state.json', '.success']:
        paths.extend(glob.glob(checkpoint_prefix + suffix))
    return paths


def copy_checkpoint(src_prefix, dst_prefix):
    src_prefix = path_to_checkpoint(src_prefix)
    paths = glob.glob(src_prefix + '.*')
    for src in paths:
        dst = re.sub(re.escape(src_prefix), dst_prefix, src, count=1)
        shutil.copyfile(src, dst)
