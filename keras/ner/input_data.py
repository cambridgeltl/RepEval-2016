from __future__ import print_function

import os
import re
import numpy
import codecs

from collections import OrderedDict, defaultdict
from logging import debug, warn
from itertools import chain, izip_longest

from common import dense_to_one_hot

class Defaults(object):
  window = 2
  max_vocab_size = None
  examples_as_indices = False

DEFAULT_W2V_FILE = 'data/w2v.bin'
DEFAULT_SENTENCE_SEPARATOR = '</s>'
DEFAULT_SENTINEL = '<ZEROS>'
DEFAULT_UNKNOWN_WORD = '<UNK>'

def load_dataset(filename, word_to_vector, label_to_index=None, config=None):
  """Load CoNLL dataset, return examples, one-hot labels and label to
  index map."""
  config = _fill_config(config, word_to_vector)
  words, labels = load_conll(filename, config)
  examples = make_examples(words, word_to_vector, config)
  features = make_word_features(words, None, config)
  label_indices, label_to_index = labels_to_indices(labels, label_to_index)
  one_hot_labels = dense_to_one_hot(label_indices, len(label_to_index))
  return words, examples, features, one_hot_labels

def _pick_word(word_to_vector, candidates, default):
  if word_to_vector is None:
    return default
  for c in candidates:
    if c in word_to_vector:
      return c
  return default

def unknown_word(config=None, word_to_vector=None):
  try:
    return config.unknown_word
  except:
    pass
  w = _pick_word(word_to_vector, ['UNKNOWN'], DEFAULT_UNKNOWN_WORD)
  warn('using %s as unknown word' % w)
  if config is not None:
    config.unknown_word = w
  return w

def sentence_separator(config=None, word_to_vector=None):
  try:
    return config.sentence_separator
  except:
    pass
  w = _pick_word(word_to_vector, ['PADDING','</s>'], DEFAULT_SENTENCE_SEPARATOR)
  warn('using %s as sentence separator' % w)
  if config is not None:
    config.sentence_separator = w
  return w

def boundary_sentinel(config=None, word_to_vector=None):
  try:
    return config.sentinel
  except:
    pass
  w = _pick_word(word_to_vector, ['<ZEROS>'], DEFAULT_SENTINEL)
  warn('using %s as boundary sentinel' % w)
  if config is not None:
    config.sentinel = w
  return w

def _fill_config(config, word_to_vector):
  if config is None:
    config = Defaults()
  try:
    examples_as_indices = config.examples_as_indices
  except:
    examples_as_indices = True
  return config

# map pattern of (previous, current, next) BIO tags to IOBES
_iobes_tag = {
    ('B', 'I', 'B'): 'E',
    ('B', 'I', 'I'): 'I',
    ('B', 'I', 'O'): 'E',
    ('I', 'I', 'B'): 'E',
    ('I', 'I', 'I'): 'I',
    ('I', 'I', 'O'): 'E',
    ('O', 'I', 'B'): 'S',
    ('O', 'I', 'I'): 'B',
    ('O', 'I', 'O'): 'S',
    ('B', 'B', 'B'): 'S',
    ('B', 'B', 'I'): 'B',
    ('B', 'B', 'O'): 'S',
    ('I', 'B', 'B'): 'S',
    ('I', 'B', 'I'): 'B',
    ('I', 'B', 'O'): 'S',
    ('O', 'B', 'B'): 'S',
    ('O', 'B', 'I'): 'B',
    ('O', 'B', 'O'): 'S',
}

def lookaround(iterable):
    "s -> (None,s0,s1), (s0,s1,s2), ..., (sn-1,sn,None), (sn,None,None)"
    a, b, c = iter(iterable), iter(iterable), iter(iterable)
    next(c, None)
    return izip_longest(chain([None], a), b, c)

def _tag(fields):
    if fields is None or len(fields) == 0:
        return 'O'
    else:
        return fields[-1][0]

def bio_to_iobes(bio_tags):
  iobes_tags = []
  def tag(t):
    return 'O' if t is None else t[0]
  for i, (prev, curr, next_) in enumerate(lookaround(bio_tags[:])):
    if curr is None:
      break
    if tag(curr) == 'O':
      iobes_tag = 'O'    # no change to out tags
    else:
      iobes_tag = _iobes_tag[(tag(prev), tag(curr), tag(next_))] + curr[1:]
    iobes_tags.append(iobes_tag)
  return iobes_tags

def load_conll(filename, config=None, encoding='utf-8'):
  """Load CoNLL-style file, return arrays (words, labels)."""
  separator = sentence_separator(config)
  try:
    separator_count = config.window if config.window else 1 # min 1
  except:
    separator_count = 1
  words, labels = [], []
  with codecs.open(filename, encoding=encoding) as f:
    for ln, line in enumerate(f, start=1):
      line = line.rstrip('\n')
      if not line:
        # sentence boundary: fill in a number of special
        # separator strings with "out" labels.
        for i in range(separator_count):
          words.append(separator)
          labels.append('O')
        continue
      fields = line.split()
      assert len(fields) == 2, 'format error on line %d: %s' % (ln, line)
      words.append(fields[0])
      labels.append(fields[1])
  try:
    if config.iobes:
      labels = bio_to_iobes(labels)
  except AttributeError:
    pass
  return numpy.array(words), numpy.array(labels)

def word_features(word, config=None):
  """Return array of surface form features for given word."""
  if word.isupper():
    return [1,0,0,0]    # all caps
  elif word[0].isupper():
    return [0,1,0,0]    # init cap
  elif any(c.isupper() for c in word):
    return [0,0,1,0]    # has cap
  else:
    return [0,0,0,1]    # no caps

def load_w2v_binary(filename, config=None):
  """Read word2vec binary format, return OrderedDict word-vector map."""

  def read_w2v_size_line(f):
    """Read word2vec file header, return (word count, vector size)."""
    l = f.readline().rstrip('\n')
    wcount, vsize = l.split()
    return int(wcount), int(vsize)

  def read_word(f):
    # http://docs.python.org/2/library/functions.html#iter
    word = ''.join(iter(lambda: f.read(1), ' '))
    return word.lstrip('\n')     # harmonize format variants

  try:
    max_rank = config.max_vocab_size
  except:
    max_rank = None

  word_to_vector = OrderedDict()
  with open(filename, 'rb') as f:
    # header has vocab and vector sizes as strings
    word_count, vec_size = map(int, f.readline().split())
    for i in range(word_count):
      if max_rank and i > max_rank:
        break
      word = read_word(f)
      vector = numpy.fromfile(f, numpy.float32, vec_size)
      features = numpy.array(word_features(word), dtype=numpy.float32)
      word_to_vector[word] = numpy.concatenate((vector, features))
  return word_to_vector

def vectors_to_examples(vectors, sentinel, config):
  if len(vectors) == 0:
    return vectors
  def vec(index):
    if index >= 0 and index < len(vectors):
      return vectors[index]
    else:
      return sentinel
  examples = []
  win = config.window
  for i, v in enumerate(vectors):
    examples.append(numpy.concatenate([vec(i+j) for j in range(-win, win+1)]))
  return numpy.array(examples)

def labels_to_indices(labels, label_to_index=None):
  """Convert labels to indices, return (indices, label_to_index map)."""
  if label_to_index is None:
    label_to_index = {}
  indices = []
  for label in labels:
    if label not in label_to_index:
      label_to_index[label] = len(label_to_index)
    indices.append(label_to_index[label])
  return numpy.array(indices), label_to_index

def make_examples(words, word_to_vector, config):
  try:
    examples_as_indices = config.examples_as_indices
  except:
    examples_as_indices = True
  if not examples_as_indices:
    mapping = word_to_vector
  else:
    # word_to_vector must be an OrderedDict for this to make sense
    mapping = { k: [i] for i, k in enumerate(word_to_vector) }
  unknown = unknown_word(config, word_to_vector)
  word_vectors = words_to_vectors(words, mapping, unknown, config)
  sentinel = boundary_sentinel(config, word_to_vector)
  return vectors_to_examples(word_vectors, mapping[sentinel], config)

def make_word_features(words, word_to_features, config):
  if word_to_features is None:
    word_to_features = {}
  features = []
  for w in words:
    if w not in word_to_features:
      word_to_features[w] = word_features(w, config)
    features.append(word_to_features[w])
  # catenate features in window with zeros at edges
  def feat(index):
    if index >= 0 and index < len(features):
      return features[index]
    else:
      return numpy.zeros(len(features[0]))
  windowed = []
  win = config.window
  for i, f in enumerate(features):
    windowed.append([feat(i+j) for j in range(-win, win+1)])
  return numpy.array(windowed)

# Penn Treebank tokenization unescapes (see
# https://www.cis.upenn.edu/~treebank/tokenization.html)
_ptb_unescape = {
  '-LRB-': '(',
  '-RRB-': ')',
  '-LSB-': '[',
  '-RSB-': ']',
  '-LCB-': '{',
  '-RCB-': '}',
  '``': '"',
  "''": '"',
}

# OntoNotes end-of-sentence punctuation escapes
_ontonotes_unescape = {
  '/.': '.',
  '/?': '?',
  '/-': '-',
}

def _unknown_word_shape(u):
  u = re.sub(r'[a-z]', 'a', u)
  u = re.sub(r'[A-Z]', 'A', u)
  u = re.sub(r'[0-9]', '0', u)
  u = re.sub(r'[^a-zA-Z0-9]', '_', u)
  return 'UNK{%s}' % u

def _normalize_word(w, word_to_vector):
  # Experimental PTB tokenization unescaping
  u = _ptb_unescape.get(w)
  if u is None or u not in word_to_vector:
    u = _ontonotes_unescape.get(w)
  if u in word_to_vector:
    if not _normalize_word.warned:
      warn('applying experimental PTB / OntoNotes token normalization')
      _normalize_word.warned = True
    return u
  w = w.lower()
  if w in word_to_vector:
    return w
  w = re.sub(r'[0-9]', '0', w)
  if w in word_to_vector:
    return w
  u = _unknown_word_shape(w)
  if u in word_to_vector:
    if not _normalize_word.unk_warned:
      warn('applying unknown token shape normalization')
      _normalize_word.unk_warned = True
    return u
  return w
_normalize_word.warned = False
_normalize_word.unk_warned = False

def words_to_vectors(words, word_to_vector, unknown_word, config=None):
  vectors = []
  for word in words:
    vector = word_to_vector.get(word)
    if vector is None:
      norm = _normalize_word(word, word_to_vector)
      vector = word_to_vector.get(norm)
      if vector is not None:
        debug('norm lookup for %s as %s' % (word, norm))
    if vector is None:
      vector = word_to_vector[unknown_word]
      words_to_vectors.oov += 1
      words_to_vectors.oov_freq[word] += 1
      debug('Out of vocabulary: %s' % word)
    vectors.append(vector)
    words_to_vectors.total += 1
  return numpy.array(vectors)
words_to_vectors.oov_freq = defaultdict(int)
words_to_vectors.total = 0
words_to_vectors.oov = 0

def oov_rate():
  return 1.*words_to_vectors.oov/words_to_vectors.total

def reset_oov():
  words_to_vectors.oov_freq = defaultdict(int)
  words_to_vectors.total = 0
  words_to_vectors.oov = 0

def most_frequent_oov(max_rank=5):
  freq_word = [(v, k) for k, v in words_to_vectors.oov_freq.items()]
  return sorted(freq_word, reverse=True)[:max_rank]

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

class DataSet(object):
  def __init__(self, words, examples, features, labels):
    assert examples.shape[0] == labels.shape[0], (
      'examples.shape: %s labels.shape: %s' % (examples.shape, labels.shape))
    self._words = words
    self._examples = examples
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._embedded = False

  @property
  def words(self):
    return self._words

  @property
  def examples(self):
    return self._examples

  @property
  def labels(self):
    return self._labels

  @property
  def word_features(self):
    return self._features

  @property
  def inputs(self):
    return [self.examples, self.word_features]

  @property
  def feature_shape(self):
    return self._features[0].shape

  @property
  def num_examples(self):
    return len(self.examples)

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def embed(self, embedding):
    assert not self._embedded, 'double embed'
    self._examples = embedding[self._examples]
    self._embedded = True

  def crop(self, max_examples):
    self._words = self._words[:max_examples]
    self._examples = self._examples[:max_examples]
    self._features = self._features[:max_examples]
    self._labels = self._labels[:max_examples]

  def subsample(self, max_size):
    step = len(self) / max_size or 1
    return DataSet(self.words[::step], self.examples[::step],
                   self.word_features[::step], self.labels[::step])

  def shuffle(self):
    permutation = numpy.arange(self.num_examples)
    numpy.random.shuffle(permutation)
    # TODO reorder also self._words
    self._examples = self._examples[permutation]
    self._features = self._features[permutation]
    self._labels = self._labels[permutation]

  def __len__(self):
    return len(self._examples)

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    assert batch_size <= self.num_examples
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self.num_examples:
      # Finished epoch
      self._epochs_completed += 1
      if shuffle:
        self.shuffle()
      else:
        warn('not shuffling between epochs')
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self.num_examples
    end = self._index_in_epoch
    return self._examples[start:end], self._labels[start:end]

class DataSets(object):
  def __init__(self, train, devel, test):
    self.train = train
    self.devel = devel
    self.test = test

  @property
  def input_size(self):
    return self.train.examples[0].shape[0]

  @property
  def feature_shape(self):
    return self.train.feature_shape

  @property
  def output_size(self):
    return self.train.labels[0].shape[0]

def _get_word_to_vector(w2v_file, config=None):
  try:
    return config.word_to_vector    # pre-loaded
  except:
    pass
  if w2v_file is None:
    w2v_file = DEFAULT_W2V_FILE
  word_to_vector = load_w2v_binary(w2v_file, config)
  wvsize = next(word_to_vector.itervalues()).shape[0]
  # If the separator, sentinel or unknown word is not defined, add
  separator = sentence_separator(config, word_to_vector)
  if separator not in word_to_vector:
    warn('no vector for sentence separator %s, adding zeros' % separator)
    word_to_vector[separator] = numpy.zeros(wvsize, dtype=numpy.float32)
  sentinel = boundary_sentinel(config, word_to_vector)
  if sentinel not in word_to_vector:
    warn('no vector for boundary sentinel %s, adding zeros' % sentinel)
    word_to_vector[sentinel] = numpy.zeros(wvsize, dtype=numpy.float32)
  unknown = unknown_word(config, word_to_vector)
  if unknown not in word_to_vector:
    warn('no vector for unknown word %s, adding zeros' % unknown)
    word_to_vector[unknown] = numpy.zeros(wvsize, dtype=numpy.float32)
  if config is not None:
    config.word_to_vector = word_to_vector
  return word_to_vector

def read_data_sets(data_dir, w2v_file=None, config=None):
  word_to_vector = _get_word_to_vector(w2v_file, config)
  # By convention, the training, development and test datasets are
  # {train,devel,test}.tsv in data_dir. The mapping from label strings
  # to indices is kept consistent over the datasets; also, make sure
  # that the label "O" maps to index 0.
  datasets = []
  label_to_index = { 'O': 0 }
  for filename in ('train.tsv', 'devel.tsv', 'test.tsv'):
    words, examples, features, labels = load_dataset(
      os.path.join(data_dir, filename), word_to_vector, label_to_index, config
    )
    datasets.append(DataSet(words, examples, features, labels))
  if oov_rate() > 0:
    warn('Out-of-vocabulary rate: %.2f%% (%d/%d)' % (100.*oov_rate(), words_to_vectors.oov, words_to_vectors.total))
    warn('top OOV: %s' % ' '.join('%s (%d)' % (w, n) for n, w in most_frequent_oov()))
  reset_oov()
  if config is not None:
    config.label_to_index = label_to_index
  return DataSets(*datasets)
