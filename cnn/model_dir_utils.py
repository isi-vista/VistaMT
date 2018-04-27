import os
import re


def find_latest_model(model_dir):
    model_files = sorted_model_files(model_dir)
    if model_files:
        return model_files[-1]
    return None


def sorted_model_files(model_dir):
    items = []
    for name in os.listdir(model_dir):
        m = re.match(r'model-iter-(\d+)\.npz$', name)
        if m:
            items.append((int(m.group(1)), name))
    items.sort()
    return [os.path.join(model_dir, x[1]) for x in items]
