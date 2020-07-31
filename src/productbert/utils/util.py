import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import sys


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def calculate_prec_rec_f1(log):
    tp = log['tp']
    fp = log['fp']
    fn = log['fn']
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print('There were only predictions for the negative class! Check your data or parameter settings!')
        precision = 'undefined'
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        print('There are no positives in the dataset!')
        sys.exit()
    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except (ZeroDivisionError, TypeError):
        print('There were only predictions for the negative class! Check your data or parameter settings!')
        f1 = 'undefined'

    return precision, recall, f1

def calculate_prec_rec_f1_multibin(log):
    tp = log['tp_multibin']
    fp = log['fp_multibin']
    fn = log['fn_multibin']
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print('There were only predictions for the negative class! Check your data or parameter settings!')
        precision = 'undefined'
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        print('There are no positives in the dataset!')
        sys.exit()
    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except (ZeroDivisionError, TypeError):
        print('There were only predictions for the negative class! Check your data or parameter settings!')
        f1 = 'undefined'

    return precision, recall, f1

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

    def total(self):
        return dict(self._data.total)
