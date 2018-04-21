
# coding: utf-8

# In[1]:


#file to read data
import glob
import random
import struct 
import csv


# In[2]:


from tensorflow.core.example import example_pb2


# In[3]:


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


# In[4]:


PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN ='[UNK]'
START_DECODING ='[START]'
STOP_DECODING ='[STOP]'


# In[5]:


class Vocab(object):
  """Vocabulary class for mapping words to integers"""


# In[6]:


def __init__(self, vocab_file, max_size):
    Args:
      vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line
      max_size: integer.
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0


# In[7]:


for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
  self._word_to_id[w] = self._count
  self._id_to_word[self._count] = w
  self._count += 1


# In[8]:


#read the file
        with open(vocab_file, 'r') as vocab_f:
          for line in vocab_f:
            pieces = line.split()
            if len(pieces) != 2:
              print 'Warning: incorrectly formatted line: %s\n' % line
              continue
            w = pieces[0]
            if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
              raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be there, but %s is' % w)
            if w in self._word_to_id:
              raise Exception('Duplicated word: %s' % w)
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            if max_size != 0 and self._count >= max_size:
              print "max_size of vocab was specified as %i; we now have %i words." % (max_size, self._count)
              break


# In[10]:


def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]


# In[11]:


def id2word(self, word_id):
    if word_id not in self._id_to_word:
        raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]


# In[12]:


def size(self):
    return self._count


# In[26]:


def write_metadata(self, fpath):
    print "Writing word embedding metadata file to %s..." % (fpath)
    with open(fpath, "w") as f:
        fieldnames = ['word']
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        for i in xrange(self.size()):
            writer.writerow({"word": self._id_to_word[i]})


# In[18]:


#main function
    def article2ids(article_words, vocab):
        ids = []
        oovs = []
        unk_id = vocab.word2id(UNKNOWN_TOKEN)
        for w in article_words:
            i = vocab.word2id(w)
            if i == unk_id:
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                ids.append(vocab.size() + oov_num)
            else:
                ids.append(i)
        return ids, oovs


# In[20]:


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids        


# In[23]:


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) 
        except ValueError as e: 
                assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary."
                article_oov_idx = i - vocab.size()
                try:
                    w = article_oovs[article_oov_idx]
                except ValueError as e:
                    raise ValueError('Error: model produced word different ID' % (i, article_oov_idx, len(article_oovs)))
                    words.append(w)
    return words      


# In[22]:


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e:
            return sents


# In[24]:


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str     


# In[25]:


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:
            if article_oovs is None:
                new_words.append("__%s__" % w)
            else:
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str


# In[ ]:




