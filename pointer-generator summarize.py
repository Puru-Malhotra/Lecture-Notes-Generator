#!/usr/bin/env python
# coding: utf-8

# In[15]:


import tensorflow as tf
import time
import os
FLAGS = tf.compat.v1.flags.Flag

def get_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config

def load_ckpt(saver, sess, ckpt_dir="train"):
 
  while True:
    try:
        latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
        ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        return ckpt_state.model_checkpoint_path
    except:
        tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
        time.sleep(10)


# In[16]:


import sys
import numpy as np


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("Usage: python inspect_checkpoint.py <file_name> \nNote: Do not include the .data .index or .meta part of the model checkpoint in file_name.")
    file_name = sys.argv[1]
    reader = tf.train.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()

    finite = []
    all_infnan = []
    some_infnan = []

    for key in sorted(var_to_shape_map.keys()):
        tensor = reader.get_tensor(key)
        if np.all(np.isfinite(tensor)):
            finite.append(key)
        else:
            if not np.any(np.isfinite(tensor)):
                all_infnan.append(key)
            else:
                some_infnan.append(key)

    print("\nFINITE VARIABLES:")
    for key in finite: print(key)

    print("\nVARIABLES THAT ARE ALL INF/NAN:")
    for key in all_infnan: print(key)

    print("\nVARIABLES THAT CONTAIN SOME FINITE, SOME INF/NAN VALUES:")
    for key in some_infnan: print(key)

    if not all_infnan and not some_infnan:
        print("CHECK PASSED: checkpoint contains no inf/NaN values")
    else:
        print("CHECK FAILED: checkpoint contains some inf/NaN values")


# In[17]:


import data

FLAGS = tf.compat.v1.flags.Flag

class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen],
                      coverage = coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch):
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[],
                     p_gens=[],
                     coverage=np.zeros([batch.enc_batch.shape[1]]) 
                     ) for _ in range(FLAGS.beam_size)]
    results = [] 

    steps = 0
    while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
        latest_tokens = [h.latest_token for h in hyps] 
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] 
        states = [h.state for h in hyps]
        prev_coverage = [h.coverage for h in hyps] 


        (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                        batch=batch,
                        latest_tokens=latest_tokens,
                        enc_states=enc_states,
                        dec_init_states=states,
                        prev_coverage=prev_coverage)

    
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps) 
        for i in range(num_orig_hyps):
            h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]  
            for j in range(FLAGS.beam_size * 2):  
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                all_hyps.append(new_hyp)

        hyps = [] 
        for h in sort_hyps(all_hyps): 
            if h.latest_token == vocab.word2id(data.STOP_DECODING):
                if steps >= FLAGS.min_dec_steps:
                    results.append(h)
            else: 
                hyps.append(h)
            if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                break

        steps += 1

    if len(results)==0:
        results = hyps

    hyps_sorted = sort_hyps(results)

    return hyps_sorted[0]

def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


# In[18]:


import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' 
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]' 
STOP_DECODING = '[STOP]' 



class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
        while True:
            filelist = glob.glob(data_path) 
            assert filelist, ('Error: Empty filelist at %s' % data_path) 
            if single_pass:
                filelist = sorted(filelist)
            else:
                random.shuffle(filelist)
            for f in filelist:
                reader = open(f, 'rb')
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes: break 
                    str_len = struct.unpack('q', len_bytes)[0]
                    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                    yield example_pb2.Example.FromString(example_str)
            if single_pass:
                print("example_generator completed reading all datafiles. No more data.")
                break


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


def abstract2ids(abstract_words, vocab, article_oovs):
    ids=[]
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx = vocab.size() + article_oovs.index(w) 
                ids.append(vocab_idx)
            else:
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) 
        except ValueError as e: 
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e: # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


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


# In[19]:


import os
#import beam_search
import json
import pyrouge
#import util
import logging
import numpy as np

FLAGS = tf.compat.v1.flags.Flag

SECS_UNTIL_NEW_CKPT = 60  


class BeamSearchDecoder(object):
    def __init__(self, model, batcher, vocab):
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = tf.train.Saver() 
        self._sess = tf.Session(config=util.get_config())

    
        ckpt_path = util.load_ckpt(self._saver, self._sess)

        if FLAGS.single_pass:
            ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] 
            self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
            if os.path.exists(self._decode_dir):
                raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

        else: 
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

    
        if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

        if FLAGS.single_pass:
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)


    def decode(self):
        t0 = time.time()
        counter = 0
        while True:
            batch = self._batcher.next_batch()  
            if batch is None: 
                assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
                tf.logging.info("Decoder has finished reading dataset for single_pass.")
                tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
                results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

            article_withunks = data.show_art_oovs(original_article, self._vocab) # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      
            best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

      
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

      
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING) 
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_output = ' '.join(decoded_words) 
 
            if FLAGS.single_pass:
               self.write_for_rouge(original_abstract_sents, decoded_words, counter) 
               counter += 1 
            else:
                print_results(article_withunks, abstract_withunks, decoded_output) 
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) 
                t1 = time.time()
                if t1-t0 > SECS_UNTIL_NEW_CKPT:
                    tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()

    def write_for_rouge(self, reference_sents, decoded_words, ex_index):
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx+1] 
            decoded_words = decoded_words[fst_period_idx+1:] 
            decoded_sents.append(' '.join(sent))

        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]


        ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
        decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

        with open(ref_file, "w") as f:
            for idx,sent in enumerate(reference_sents):
                f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
        with open(decoded_file, "w") as f:
            for idx,sent in enumerate(decoded_sents):
                f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

        tf.logging.info("Wrote example %i to file" % ex_index)


    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        article_lst = article.split() 
        decoded_lst = decoded_words 
        to_write = {
            'article_lst': [make_html_safe(t) for t in article_lst],
            'decoded_lst': [make_html_safe(t) for t in decoded_lst],
            'abstract_str': make_html_safe(abstract),
            'attn_dists': attn_dists
        }
        if FLAGS.pointer_gen:
            to_write['p_gens'] = p_gens
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)
        tf.logging.info('Wrote visualization data to %s', output_fname)


    def print_results(article, abstract, decoded_output):
        print("---------------------------------------------------------------------------")
        tf.logging.info('ARTICLE:  %s', article)
        tf.logging.info('REFERENCE SUMMARY: %s', abstract)
        tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
        print("---------------------------------------------------------------------------")


    def make_html_safe(s):
        s.replace("<", "&lt;")
        s.replace(">", "&gt;")
        return s


    def rouge_eval(ref_dir, dec_dir):
        r = pyrouge.Rouge155()
        r.model_filename_pattern = '#ID#_reference.txt'
        r.system_filename_pattern = '(\d+)_decoded.txt'
        r.model_dir = ref_dir
        r.system_dir = dec_dir
        logging.getLogger('global').setLevel(logging.WARNING) 
        rouge_results = r.convert_and_evaluate()
        return r.output_to_dict(rouge_results)


    def rouge_log(results_dict, dir_to_write):
        log_str = ""
        for x in ["1","2","l"]:
            log_str += "\nROUGE-%s:\n" % x
            for y in ["f_score", "recall", "precision"]:
                key = "rouge_%s_%s" % (x,y)
                key_cb = key + "_cb"
                key_ce = key + "_ce"
                val = results_dict[key]
                val_cb = results_dict[key_cb]
                val_ce = results_dict[key_ce]
                log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
        tf.logging.info(log_str) # log to screen
        results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
        tf.logging.info("Writing final ROUGE results to %s...", results_file)
        with open(results_file, "w") as f:
            f.write(log_str)

    def get_decode_dir_name(ckpt_name):
        if "train" in FLAGS.data_path: dataset = "train"
        elif "val" in FLAGS.data_path: dataset = "val"
        elif "test" in FLAGS.data_path: dataset = "test"
        else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
        dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
        if ckpt_name is not None:
            dirname += "_%s" % ckpt_name
        return dirname


# In[20]:


import queue as Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data


class Example(object):
    def __init__(self, article, abstract_sentences, vocab, hps):
        self.hps = hps

        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        article_words = article.split()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        self.enc_len = len(article_words) 
        self.enc_input = [vocab.word2id(w) for w in article_words] 
    
        abstract = ' '.join(abstract_sentences) 
        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words] 

        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

    
        if hps.pointer_gen:
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

    
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: 
            inp = inp[:max_len]
            target = target[:max_len] 
        else: 
            target.append(stop_id) 
        assert len(inp) == len(target)
        return inp, target


    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    def __init__(self, example_list, hps, vocab):
        self.pad_id = vocab.word2id(data.PAD_TOKEN) 
        self.init_encoder_seq(example_list, hps) 
        self.init_decoder_seq(example_list, hps) 
        self.store_orig_strings(example_list) 

    def init_encoder_seq(self, example_list, hps):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

    def init_decoder_seq(self, example_list, hps):
                                    for ex in example_list:
                                                     ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

                                    self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
                                    self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
                                    self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
                                    for i, ex in enumerate(example_list):
                                          self.dec_batch[i, :] = ex.dec_input[:]
                                          self.target_batch[i, :] = ex.target[:]
                                          for j in range(ex.dec_len):
                                                                   self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
                                    self.original_articles = [ex.original_article for ex in example_list] 
                                    self.original_abstracts = [ex.original_abstract for ex in example_list] 
                                    self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] 


class Batcher(object):
    BATCH_QUEUE_MAX = 100 

    def __init__(self, data_path, vocab, hps, single_pass):
                                    self._data_path = data_path
                                    self._vocab = vocab
                                    self._hps = hps
                                    self._single_pass = single_pass
                                    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
                                    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)
                                    if single_pass:
                                          self._num_example_q_threads = 1 
                                          self._num_batch_q_threads = 1  
                                          self._bucketing_cache_size = 1 
                                          self._finished_reading = False 
                                    else:
                                          self._num_example_q_threads = 16 
                                          self._num_batch_q_threads = 4  
                                          self._bucketing_cache_size = 100 
    
                                    self._example_q_threads = []
                                    for _ in range(self._num_example_q_threads):
                                          self._example_q_threads.append(Thread(target=self.fill_example_queue))
                                          self._example_q_threads[-1].daemon = True
                                          self._example_q_threads[-1].start()
                                    self._batch_q_threads = []
                                    for _ in range(self._num_batch_q_threads):
                                          self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
                                          self._batch_q_threads[-1].daemon = True
                                          self._batch_q_threads[-1].start()

                                    if not single_pass:
                                          self._watch_thread = Thread(target=self.watch_threads)
                                          self._watch_thread.daemon = True
                                          self._watch_thread.start()


    def next_batch(self):
                                    if self._batch_queue.qsize() == 0:
                                           tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
                                           if self._single_pass and self._finished_reading:
                                                                        tf.logging.info("Finished reading dataset in single_pass mode.")
                                                                        return None

                                    batch = self._batch_queue.get() 
                                    return batch

    def fill_example_queue(self):
                                    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

                                    while True:
                                      try:
                                        (article, abstract) = next(input_gen) 
                                      except StopIteration: 
                                        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                                        if self._single_pass:
                                            tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                                            self._finished_reading = True
                                            break
                                        else:
                                            raise Exception("single_pass mode is off but the example generator is out of data; error.")
                                        abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] 
                                        example = Example(article, abstract_sentences, self._vocab, self._hps) 
                                        self._example_queue.put(example) 


    def fill_batch_queue(self):
                                    while True:
                                      if self._hps.mode != 'decode':
                                        inputs = []
                                        for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                                                    inputs.append(self._example_queue.get())
                                        inputs = sorted(inputs, key=lambda inp: inp.enc_len)
                                        batches = []
                                        for i in range(0, len(inputs), self._hps.batch_size):
                                            batches.append(inputs[i:i + self._hps.batch_size])
                                        if not self._single_pass:
                                            shuffle(batches)
                                        for b in batches:  
                                            self._batch_queue.put(Batch(b, self._hps, self._vocab))

                                    else: 
                                            ex = self._example_queue.get()
                                            b = [ex for _ in range(self._hps.batch_size)]
                                            self._batch_queue.put(Batch(b, self._hps, self._vocab))


    def watch_threads(self):
                                       
        while True:
          time.sleep(60)
          for idx,t in enumerate(self._example_q_threads):
            if not t.is_alive(): 
                tf.logging.error('Found example queue thread dead. Restarting.')
                new_t = Thread(target=self.fill_example_queue)
                self._example_q_threads[idx] = new_t
                new_t.daemon = True
                new_t.start()
                
          for idx,t in enumerate(self._batch_q_threads):
            if not t.is_alive(): 
                tf.logging.error('Found batch queue thread dead. Restarting.')
                new_t = Thread(target=self.fill_batch_queue)
                self._batch_q_threads[idx] = new_t
                new_t.daemon = True
                new_t.start()


    def text_generator(self, example_generator):
        while True:
            e = next(example_generator) 
            try:
                article_text = e.features.feature['article'].bytes_list.value[0].decode() 
                abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode() 
            except ValueError:
                  tf.logging.error('Failed to get article or abstract from example')
                  continue
            if len(article_text)==0:
                  tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                  yield (article_text, abstract_text)


# In[21]:


import os
import time
#from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.compat.v1.flags.Flag

class SummarizationModel(object):
    
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        hps = self._hps
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

        if hps.mode=="decode" and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')


    def _make_feed_dict(self, batch, just_enc=False):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        if FLAGS.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs) 
        return encoder_outputs, fw_st, bw_st


    def _reduce_states(self, fw_st, bw_st):
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
                        
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) 
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) 
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) 
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) 
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) 


    def _add_decoder(self, inputs):
        hps = self._hps
        cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None 
        outputs, out_state, attn_dists, p_gens, coverage = tf.contrib.legacy_seq2seq.attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(hps.mode=="decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=prev_coverage)

        return outputs, out_state, attn_dists, p_gens, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists):
        with tf.variable_scope('final_distribution'):
            vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]

            extended_vsize = self._vocab.size() + self._max_art_oovs 
            extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] 
            batch_nums = tf.range(0, limit=self._hps.batch_size) 
            batch_nums = tf.expand_dims(batch_nums, 1) 
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1] 
            batch_nums = tf.tile(batch_nums, [1, attn_len]) 
            indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) 
            shape = [self._hps.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] 
            final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        train_dir = os.path.join(FLAGS.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        self._vocab.write_metadata(vocab_metadata_path) 
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.size() 

        with tf.variable_scope('seq2seq'):
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                if hps.mode=="train": self._add_emb_vis(embedding) 
                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) 
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] 

          
            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self._enc_states = enc_outputs

            self._dec_in_state = self._reduce_states(fw_st, bw_st)

            with tf.variable_scope('decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)

      
            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                w_t = tf.transpose(w)
                v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = [] 
                for i,output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) 

                vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] 


      
        if FLAGS.pointer_gen:
            final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
        else: 
            final_dists = vocab_dists



        if hps.mode in ['train', 'eval']:
            with tf.variable_scope('loss'):
                if FLAGS.pointer_gen:
                    loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
                    batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
                    for dec_step, dist in enumerate(final_dists):
                        targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
                        indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
                        gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
                        losses = -tf.log(gold_probs)
                        loss_per_step.append(losses)

            
                    self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                else:
                    self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

                tf.summary.scalar('loss', self._loss)

          
                if hps.coverage:
                    with tf.variable_scope('coverage_loss'):
                        self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                        tf.summary.scalar('coverage_loss', self._coverage_loss)
                    self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
                    tf.summary.scalar('total_loss', self._total_loss)

        if hps.mode == "decode":
            assert len(final_dists)==1 
            final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) 
            self._topk_log_probs = tf.log(topk_probs)


    def _add_train_op(self):
        loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

        
        tf.summary.scalar('global_norm', global_norm)

        
        optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


    def build_graph(self):
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        with tf.device("/gpu:0"):
            self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        feed_dict = self._make_feed_dict(batch, just_enc=True) 
        (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict)

        
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        return enc_states, dec_in_state


    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
        beam_size = len(dec_init_states)

        
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  
        new_h = np.concatenate(hiddens, axis=0)  
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
        }

        to_return = {
          "ids": self._topk_ids,
          "probs": self._topk_log_probs,
          "states": self._dec_out_state,
          "attn_dists": self.attn_dists
        }

        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed) 

        
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]

        
        assert len(results['attn_dists'])==1
        attn_dists = results['attn_dists'][0].tolist()

        if FLAGS.pointer_gen:
            assert len(results['p_gens'])==1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
    dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
    values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
    coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss


# In[28]:


import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
#from data import Vocab
#from batcher import Batcher
#from model import SummarizationModel
#from decode import BeamSearchDecoder
#import util
from tensorflow.python import debug as tf_debug

FLAGS = tf.compat.v1.flags.Flag

# Where to find data
tf.compat.v1.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.compat.v1.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.compat.v1.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.compat.v1.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.compat.v1.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.compat.v1.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.compat.v1.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.compat.v1.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.compat.v1.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.compat.v1.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.compat.v1.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.compat.v1.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.compat.v1.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.compat.v1.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.compat.v1.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.compat.v1.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.compat.v1.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.compat.v1.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.compat.v1.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.compat.v1.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.compat.v1.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.compat.v1.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.compat.v1.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.compat.v1.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.compat.v1.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")



def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
          running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print ("Restored %s." % curr_ckpt)
     # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print ("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print ("Saved.")
    exit()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())
    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
 
    model.build_graph() # build the graph
    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    if FLAGS.restore_best_model:
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

    sv = tf.train.Supervisor(logdir=train_dir,
                       is_chief=True,
                       saver=saver,
                       summary_op=None,
                       save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                       save_model_secs=60, # checkpoint every 60 secs
                       global_step=model.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info("Created session.")
    try:
        run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
  
    tf.logging.info("starting run_training")
    with sess_context_manager as sess:
        if FLAGS.debug: # start the tensorflow debugger
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        while True: # repeats until interrupted
            batch = batcher.next_batch()

            tf.logging.info('running training step...')
            t0=time.time()
            results = model.run_train_step(sess, batch)
            t1=time.time()
            tf.logging.info('seconds for training step: %.3f', t1-t0)

            loss = results['loss']
            tf.logging.info('loss: %f', loss) # print the loss to screen

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            if FLAGS.coverage:
                coverage_loss = results['coverage_loss']
                tf.logging.info("coverage_loss: %f", coverage_loss) # print the coverage loss to screen

            # get the summaries and iteration number so we can write summaries to tensorboard
            summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
            train_step = results['global_step'] # we need this to update our running average loss

            summary_writer.add_summary(summaries, train_step) # write the summaries
            if train_step % 100 == 0: # flush the summary writer every so often
                summary_writer.flush()


def run_eval(model, batcher, vocab):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    model.build_graph() # build the graph
    saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess) # load a new checkpoint
        batch = batcher.next_batch() # get the next batch

        # run eval on the batch
        t0=time.time()
        results = model.run_eval_step(sess, batch)
        t1=time.time()
        tf.logging.info('seconds for batch: %.2f', t1-t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        tf.logging.info('loss: %f', loss)
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()


def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode=="train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
    hps_dict = {}
    for key,val in FLAGS.__flags.items(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111) # a seed value for randomness

    if hps.mode == 'train':
        print("creating model...")
        model = SummarizationModel(hps, vocab)
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps  # This will be the hyperparameters for the decoder model
        decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
    tf.compat.v1.run()


# In[ ]:




