import os
import re


MODEL_PREFIX = 'model-iter-'


def find_latest_model(model_dir):
    model_files = sorted_model_files(model_dir)
    if model_files:
        return model_files[-1]
    return None


def model_iter_from_path(path):
    m = re.match(r'{}(\d+)\.npz$'.format(MODEL_PREFIX), os.path.basename(path))
    model_iter = None
    if m:
        model_iter = int(m.group(1))
    return model_iter


def sorted_model_files(model_dir):
    items = []
    for name in os.listdir(model_dir):
        model_iter = model_iter_from_path(name)
        if model_iter:
            items.append((model_iter, name))
    items.sort()
    return [os.path.join(model_dir, x[1]) for x in items]
