import os
import sys
import logging
import numpy as np

from collections import namedtuple
from datetime import datetime
from errno import EEXIST
from logging import warn

import conlleval

try:
  import tensorflow as tf
  WITH_TF = True
except ImportError:
  WITH_TF = False

BinaryClassificationCounts = namedtuple('BinaryClassificationCounts',
                                        'tp tn fp fn')
BinaryClassificationMetrics = namedtuple('BinaryClassificationMetrics',
                                         'tp tn fp fn acc prec rec fscore')

LOGDIR = 'logs'
PREDDIR = 'predictions'

# Label for "out" class in tagging scheme (IOB, IOBES, etc.)
OUT_LABEL = 'O'

def flatten(t):
  """Reshape tensor [None, d1, d2, ...] to [None, d1 * d2 * ...]."""
  if not WITH_TF:
    raise NotImplementedError
  return tf.reshape(t, [-1, np.prod(t.get_shape().as_list()[1:])])

def tsize(t):
  """Return the number of elements in tensor, ignoring any None components."""
  return np.prod([d for d in t.get_shape().as_list() if d is not None])

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

def dense_to_one_hot(dense, num_distinct=None):
  """Convert index values to one-hot vectors."""
  if num_distinct is None:
    num_distinct = len(uniq(dense.ravel()))
  size = dense.shape[0]
  if size == 0:
    return np.array([])
  hot_indices = np.arange(size) * num_distinct + dense.ravel()
  one_hot = np.zeros((size, num_distinct))
  one_hot.flat[hot_indices] = 1
  return one_hot

def one_hot_to_dense(one_hot):
    """Convert one-hot vectors to index values."""
    return np.argmax(one_hot, axis=1)

def one_hot_to_label(one_hot, index_to_label):
    """Convert one-hot vectors to labels."""
    indices = one_hot_to_dense(one_hot)
    return [index_to_label[i] for i in indices]

def word_to_vector_to_matrix(word_to_vector):
    """Convert word to vector mapping to embedding matrix."""
    vectors = word_to_vector.values()
    # Add a zero vector for OOV marker and bounary sentinel
    vectors.append(np.zeros(vectors[0].shape, dtype=vectors[0].dtype))
    return np.array(vectors)

def write_gold_and_prediction(dataset, predictions, config, out=None):
  """Write out TAB-separated (word, gold-label, predicted-label) triples."""
  if out is None:
    out = sys.stdout
  index_to_label = { v: k for k, v in config.label_to_index.items() }
  gold = one_hot_to_label(dataset.labels, index_to_label)
  pred = [index_to_label[i] for i in predictions]
  assert len(pred) == len(dataset.words)
  for word, label, prediction in zip(dataset.words, gold, pred):
    try:
      print >> out, '%s\t%s\t%s' % (word.encode('utf-8'), label, prediction)
    except:
      warn('Error writing  %s, writing <ERROR> instead', word)
      print >> out, '<ERROR>\t%s\t%s' % (label, prediction)

def accuracy(gold, pred):
    """Return accuracy for gold and predicted label values."""
    #return 1.*sum(1 for g, p in zip(gold, predicted) if g == p) / len(gold)
    return np.average(np.equal(gold, pred))

def tp_tn_fp_fn(gold, predicted):
    """Return (TP, FN, FP, FN) counts for gold and prediced label values.

    Assumes that 0 is negative and all others positive.
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for g, p in zip(gold, predicted):
        if g == p:
            if g == 0:
                tn += 1
            else:
                tp += 1
        else:
            if g == 0:
                fp += 1
            else:
                fn += 1
    return BinaryClassificationCounts(tp, tn, fp, fn)

def precision_recall_fscore(tp, fp, fn):
    """Return (precision, recall, f-score) for given counts."""
    prec = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    rec = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f

def classification_summary(gold, pred):
    """Return string summarizing binary classification performance."""
    a = accuracy(gold, pred)
    # Binarize for p/r/f
    gold = (np.asarray(gold) != 0).astype(int)
    pred = (np.asarray(pred) != 0).astype(int)
    tp, tn, fp, fn = tp_tn_fp_fn(gold, pred)
    p, r, f = precision_recall_fscore(tp, fp, fn)
    return '%.2f%% a %.2f%% f (%.1fp %.1fr %dtp %dfp %dfn)' % (
        100.*a, 100.*f, 100.*p, 100.*r, tp, fp, fn) #, tn)

def tag_type(label):
  """Return type name for IOB-like label or None if out or not an IOB label."""
  if label == OUT_LABEL or label[:2] not in ('B-', 'I-', 'E-', 'S-'):
    return None
  else:
    return label[2:]

def iob_types(labels):
  """Return type names for IOB-like labels."""
  return set(tag_type(l) for l in labels if l != OUT_LABEL)

def is_iob_labeling(labels):
  """Return True if given labels are an IOB-like labeling, False otherwise."""
  if OUT_LABEL not in labels:
    return False
  else:
    return all(tag_type(l) for l in labels if l != OUT_LABEL)

def binarize(a, positive=None):
  """Return values mapped to 1 or 0.

  If positive is not None, map values in positive to 1 and others to 0,
  otherwise map all non-zero labels to 1.
  """
  if positive is None:
    return (np.asarray(a) != 0).astype(int)
  else:
    if not isinstance(positive, (np.ndarray, list)):
      positive = np.array(list(positive))    # handle arbitrary iterables
    if any(not isinstance(p, (int, long, float, complex)) for p in positive):
      warn('binarize:positive contains non-number')
    return (np.in1d(a, positive)).astype(int)

def as_dense(a):
  """Return dense equivalent of (optionally) one-hot array."""
  a = np.asarray(a)
  if len(a.shape) == 1:
    return a    # already dense
  elif len(a.shape) == 2:
    return np.argmax(a, axis=1)
  else:
    raise ValueError('cannot map array of shape %s' % str(a.shape))

def evaluate_binary_classification(gold, pred, positive=None):
  """Evaluate binary classification performance.

  If positive is not None, map labels in positive to 1 and others
  to 0. Otherwise map all non-zero labels to 1.

  Return BinaryClassificationMetrics.
  """
  gold = as_dense(gold)
  pred = as_dense(pred)

  if len(gold) != len(pred):
    raise ValueError('labels counts do not match')

  gold = binarize(gold, positive)
  pred = binarize(pred, positive)

  if not np.any(gold):
    warn('no positive gold labels for %s' % str(positive))

  acc = accuracy(gold, pred)
  tp, tn, fp, fn = tp_tn_fp_fn(gold, pred)
  prec, rec, f = precision_recall_fscore(tp, fp, fn)

  return BinaryClassificationMetrics(tp, tn, fp, fn, acc, prec, rec, f)

def group_labels(labels):
  """Group label strings into sets of equivalents.

  For IOB-like labels groups together "in" labels (B-, I-, etc.) by
  type.

  Return list of (group-name, positive-labels) tuples.
  """
  if not is_iob_labeling(labels):
    return [(l, [l]) for l in labels]    # no groups
  else:
    types = iob_types(labels)
    return [(t, [l for l in labels if l.endswith('-'+t)]) for t in types]

def average_precision_recall_fscore(results, micro=True):
  """Return average precision, recall and f-score for list of
  BinaryClassificationMetrics.
  """
  if micro:
    total = BinaryClassificationMetrics(*tuple(np.sum(results, axis=0)))
    return precision_recall_fscore(total.tp, total.fp, total.fn)
  else:
    avg = BinaryClassificationMetrics(*tuple(np.average(results, axis=0)))
    return avg.prec, avg.rec, avg.fscore

def per_type_summary(gold, pred, config):
  """Return string summarizing per-class classification performance."""
  indices = np.unique([as_dense(gold), as_dense(pred)])
  index_to_label = { v: k for k, v in config.label_to_index.items() }
  labels = [index_to_label[i] for i in indices]

  results = {}
  for name, positive in group_labels(labels):
    pos = map(lambda l: config.label_to_index[l], positive)
    results[name] = evaluate_binary_classification(gold, pred, pos)

  summaries = []
  nlen = max(len(name) for name, _ in group_labels(labels))
  for name, r in results.items():
    summaries.append('%*s %.2f%% f (%.1fp %.1fr %dtp %dfp %dfn)' % (
      nlen, name, 100.*r.fscore, 100.*r.prec, 100.*r.rec, r.tp, r.fp, r.fn
    ))

  acc = accuracy(gold, pred)
  _, _, microf = average_precision_recall_fscore(results.values(), micro=True)
  _, _, macrof = average_precision_recall_fscore(results.values(), micro=False)
  summaries.append('%.2f%% acc %.2f%% micf %.2f%% macf' % (
    100.*acc, 100.*microf, 100.*macrof
  ))

  return '\n'.join(summaries)

def conll_summary(tokens, gold, pred, config):
  """Return string summarizing performance using CoNLL criteria."""
  index_to_label = { v: k for k, v in config.label_to_index.items() }

  acc = accuracy(gold, pred)
  gold = map(lambda i: index_to_label[i], as_dense(gold))
  pred = map(lambda i: index_to_label[i], as_dense(pred))

  # Format as space-separated (token, gold, pred) strings for CoNLL eval.
  if len(tokens) != len(gold) or len(gold) != len(pred):
    raise ValueError('counts do not match')
  formatted = [' '.join(t) for t in zip(tokens, gold, pred)]

  o, by_type = conlleval.metrics(conlleval.evaluate(formatted))
  nlen = max(len(name) for name in by_type.keys())
  summaries = ['%.2f%% acc %.2f%% f (%.1fp %.1fr %dtp %dfp %dfn)' % (
    100.*acc, 100.*o.fscore, 100.*o.prec, 100.*o.rec,  o.tp, o.fp, o.fn
  )]
  for name, r in sorted(by_type.items()):
    summaries.append('%*s %.2f%% f (%.1fp %.1fr %dtp %dfp %dfn)' % (
      nlen, name, 100.*r.fscore, 100.*r.prec, 100.*r.rec, r.tp, r.fp, r.fn
    ))

  return '\n'.join(summaries)

def performance_summary(tokens, gold, pred, config):
  """Return string summarizing performance using appropriate metrics."""
  # Mention-level (CoNLL) evaluation by default for IOB-like labels,
  # tag-level evaluation for others; config option can override
  try:
    token_level_eval = config.token_level_eval
  except AttributeError:
    token_level_eval = False
  if not token_level_eval and is_iob_labeling(config.label_to_index.keys()):
    return conll_summary(tokens, gold, pred, config)
  else:
    return per_type_summary(gold, pred, config)

def safe_makedirs(path):
  """Create directory path if it doesn't already exist."""
  # From http://stackoverflow.com/a/5032238
  try:
    os.makedirs(path)
    logging.warn('Created directory %s/' % path)
  except OSError, e:
    if e.errno != EEXIST:
      raise

def _logname(name, suffix='.log'):
  if _logname.timestr is None:
    # Remember first invocation time in format 2015-12-31--23-59-59--999999
    ts = datetime.now().isoformat('_')
    _logname.timestr = ts.replace(':','-').replace('_','--').replace('.','--')
  return name + '--' + _logname.timestr + suffix
_logname.timestr = None

def setup_logging(name=None):
  # Configure logging to stderr and file.
  if name is None:
    name = 'root'
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  # logging to stderr
  sh = logging.StreamHandler(sys.stderr)
  sh.setLevel(logging.INFO)
  sh.setFormatter(logging.Formatter(name + ':%(message)s'))
  logger.addHandler(sh)
  # logging to file
  safe_makedirs(LOGDIR)
  timestr = datetime.now().isoformat('_').split('.')[0].replace(':','.')
  fh = logging.FileHandler(os.path.join(LOGDIR, _logname(name)))
  fh.setLevel(logging.INFO)
  fh.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
  logger.addHandler(fh)

def save_gold_and_prediction(dataset, predictions, config, name):
  safe_makedirs(PREDDIR)
  with open(os.path.join(PREDDIR, _logname(name, '.txt')), 'wt') as out:
    write_gold_and_prediction(dataset, predictions, config, out)
