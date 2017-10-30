import lasagne
import numpy as np
import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
import os
import json
import os
#from random import shuffle
import random

def load_data(in_file, max_example=None, relabeling=True, question_belong=[]):
    documents = []
    questions = []
    answers = []
    options = []
    num_examples = 0
    def get_file(path):
        files = []
        for inf in os.listdir(path):
            new_path = os.path.join(path, inf)
            if os.path.isdir(new_path):
                assert inf in ["middle", "high"]
                files += get_file(new_path)
            else:
                if new_path.find(".DS_Store") != -1:
                    continue
                files += [new_path]
        return files
    files = get_file(in_file)
    for inf in files:
        obj = json.load(open(inf, "r"))
        for i, q in enumerate(obj["questions"]):
            question_belong += [inf + "_" + str(i)]
            documents += [obj["article"]]
            questions += [q]
            assert len(obj["options"][i]) == 4
            options += obj["options"][i]
            answers += [ord(obj["answers"][i]) - ord('A')]
            num_examples += 1
        if (max_example is not None) and (num_examples >= max_example):
            break
    def clean(st_list):
        for i, st in enumerate(st_list):
            st_list[i] = st.lower().strip()
        return st_list
    documents = clean(documents)
    questions = clean(questions)
    options = clean(options)
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, options, answers)

def load_data_classify(in_file, max_example=None, relabeling=True, question_belong=[]):

    documents = []
    questions = []
    answers = []
    options = []
    num_examples = 0
    title_doc = []
    title_q = []
    title_ops = []
    title_a = []
    false_doc=[]
    false_q=[]
    false_ops=[]
    false_a=[]
    how_quant_doc=[]
    how_quant_q=[]
    how_quant_ops=[]
    how_quant_a=[]
    where_doc=[]
    where_q=[]
    where_ops=[]
    where_a=[]
    when_doc=[]
    when_q=[]
    when_ops=[]
    when_a=[]
    why_doc=[]
    why_q=[]
    why_ops=[]
    why_a=[]
    who_doc,who_q,who_ops,who_a=[],[],[],[]
    un_doc,un_q,un_ops,un_a=[],[],[],[]
    mean_doc,mean_q,mean_ops,mean_a=[],[],[],[]
    which_doc,which_q,which_ops,which_a=[],[],[],[]
    blank_doc,blank_q,blank_ops,blank_a=[],[],[],[]
    what_doc,what_q,what_ops,what_a=[],[],[],[]
    story_doc,story_q,story_ops,story_a=[],[],[],[]
    def get_file(path):
        files = []
        for inf in os.listdir(path):
            new_path = os.path.join(path, inf)
            if os.path.isdir(new_path):
                assert inf in ["middle", "high"]
                files += get_file(new_path)
            else:
                if new_path.find(".DS_Store") != -1:
                    continue
                files += [new_path]
        return files
    files = get_file(in_file)
    for inf in files:
        obj = json.load(open(inf, "r"))
        for i, q in enumerate(obj["questions"]):
            q=q.lower()
            if "title" in q:
               title_doc += [obj["article"]]
               title_q += [q]
               title_ops +=obj["options"][i]
               title_a += [ord(obj["answers"][i])-ord('A')]
            elif "not true" in q or "false" in q or "not right" in q or "not correct" in q:
               if "not true except" not in q and "not false" not in q and "false except" not in q:
                false_doc += [obj["article"]]
                false_q += [q]
                false_ops +=obj["options"][i]
                false_a += [ord(obj["answers"][i])-ord('A')]
            elif "mean" in q:
               mean_doc += [obj["article"]]
               mean_q += [q]
               mean_ops +=obj["options"][i]
               mean_a += [ord(obj["answers"][i])-ord('A')]
            elif "story" in q:
               story_doc += [obj["article"]]
               story_q += [q]
               story_ops +=obj["options"][i]
               story_a += [ord(obj["answers"][i])-ord('A')]
            elif "how many" in q or "how long" in q or "how much" in q or "how old" in q:
               how_quant_doc += [obj["article"]]
               how_quant_q += [q]
               how_quant_ops +=obj["options"][i]
               how_quant_a += [ord(obj["answers"][i])-ord('A')]
            elif "where" in q:
               where_doc += [obj["article"]]
               where_q += [q]
               where_ops +=obj["options"][i]
               where_a += [ord(obj["answers"][i])-ord('A')]
            elif "when" in q:
               when_doc += [obj["article"]]
               when_q += [q]
               when_ops +=obj["options"][i]
               when_a += [ord(obj["answers"][i])-ord('A')]
            elif "why" in q:
               why_doc += [obj["article"]]
               why_q += [q]
               why_ops +=obj["options"][i]
               why_a += [ord(obj["answers"][i])-ord('A')]
            elif "who" in q:
               who_doc += [obj["article"]]
               who_q += [q]
               who_ops +=obj["options"][i]
               who_a += [ord(obj["answers"][i])-ord('A')]
            elif "which" in q:
               which_doc += [obj["article"]]
               which_q += [q]
               which_ops +=obj["options"][i]
               which_a += [ord(obj["answers"][i])-ord('A')]
            elif "what" in q:
               what_doc += [obj["article"]]
               what_q += [q]
               what_ops +=obj["options"][i]
               what_a += [ord(obj["answers"][i])-ord('A')]
            elif "_" in q:
               blank_doc += [obj["article"]]
               blank_q += [q]
               blank_ops +=obj["options"][i]
               blank_a += [ord(obj["answers"][i])-ord('A')]
            else:
               un_doc += [obj["article"]]
               un_q += [q]
               un_ops +=obj["options"][i]
               un_a += [ord(obj["answers"][i])-ord('A')]
            question_belong += [inf + "_" + str(i)]
            documents += [obj["article"]]
            questions += [q]
            assert len(obj["options"][i]) == 4
            options += obj["options"][i]
            answers += [ord(obj["answers"][i]) - ord('A')]
            num_examples += 1
        if (max_example is not None) and (num_examples >= max_example):
            break
    def clean(st_list):
        for i, st in enumerate(st_list):
            st_list[i] = st.lower().strip()
        return st_list
    documents = clean(documents)
    questions = clean(questions)
    options = clean(options)
    logging.info('#Examples: %d' % len(documents))
    where_all=[clean(where_doc),clean(where_q),clean(where_ops),where_a]
    how_quant_all=[clean(how_quant_doc),clean(how_quant_q),clean(how_quant_ops),how_quant_a]
    when_all=[clean(when_doc),clean(when_q),clean(when_ops),when_a]
    why_all=[clean(why_doc),clean(why_q),clean(why_ops),why_a]
    title_all=[clean(title_doc),clean(title_q),clean(title_ops),title_a]
    false_all=[clean(false_doc),clean(false_q),clean(false_ops),false_a]
    who_all=[clean(who_doc),clean(who_q),clean(who_ops),who_a]
    mean_all=[clean(mean_doc),clean(mean_q),clean(mean_ops),mean_a]
    story_all=[clean(story_doc),clean(story_q),clean(story_ops),story_a]
    which_all=[clean(which_doc),clean(which_q),clean(which_ops),which_a]
    what_all=[clean(what_doc),clean(what_q),clean(what_ops),what_a]
    blank_all=[clean(blank_doc),clean(blank_q),clean(blank_ops),blank_a]
    un_all=[clean(un_doc),clean(un_q),clean(un_ops),un_a]
    return (documents, questions, options, answers), title_all, false_all,mean_all,story_all,how_quant_all, where_all,when_all,why_all,who_all,which_all,what_all,blank_all,un_all


def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    word_count['a'] = 100000
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}
'''
def vectorize(examples, word_dict,
              sort_by_len=True, verbose=True, concat=False):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_x3 = []
    in_y = []
    def get_vector(st):
        seq = [word_dict[w] if w in word_dict else 0 for w in st]
        return seq

    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[3])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        assert 0 <= a <= 3
        seq1 = get_vector(d_words)
        seq2 = get_vector(q_words)
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1 += [seq1]
            in_x2 += [seq2]
            option_seq = []
            for i in range(4):
                if concat:
                    op = " ".join(q_words) + ' @ ' + examples[2][i + idx * 4]
                else:
                    op = examples[2][i + idx * 4]
                op = op.split(' ')
                option = get_vector(op)
                assert len(option) > 0
                option_seq += [option]
            in_x3 += [option_seq]
            in_y.append(a)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[0])))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_y = [in_y[i] for i in sorted_index]
        in_x3 = [in_x3[i] for i in sorted_index]
    new_in_x3 = []
    for i in in_x3:
        #print i
        new_in_x3 += i
    #print new_in_x3
    return in_x1, in_x2, new_in_x3, in_y
'''

def vectorize(examples, word_dict,
              sort_by_len=True, verbose=True, concat=False):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_x3 = []
    in_y = []
    def get_vector(st):
        seq = [word_dict[w] if w in word_dict else 0 for w in st]
        return seq
    options = zip(*[iter(examples[2])]*4)
    options = [list(option) for option in options]
    exs = zip(examples[0], examples[1], options, examples[3])
    #shuffle(exs)
    random.Random(1234).shuffle(exs)
    squery = []
    sdoc = []
    sops = []
    sans = []
    for ex in exs:
        sdoc.append(ex[0])
        squery.append(ex[1])
        sans.append(ex[3])
        for op in ex[2]:
            sops.append(op)
    ret=[sdoc,squery,sops,sans]

    for idx, (d, q, a) in enumerate(zip(sdoc, squery, sans)):
        d_words = d.split(' ')
        q_words = q.split(' ')
        assert 0 <= a <= 3
        seq1 = get_vector(d_words)
        seq2 = get_vector(q_words)
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1 += [seq1]
            in_x2 += [seq2]
            option_seq = []
            for i in range(4):
                if concat:
                    op = " ".join(q_words) + ' @ ' + sops[i + idx * 4]
                else:
                    op = sops[i + idx * 4]
                op = op.split(' ')
                option = get_vector(op)
                assert len(option) > 0
                option_seq += [option]
            in_x3 += [option_seq]
            in_y.append(a)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(sdoc)))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_y = [in_y[i] for i in sorted_index]
        in_x3 = [in_x3[i] for i in sorted_index]
    new_in_x3 = []
    for i in in_x3:
        #print i
        new_in_x3 += i
    #print new_in_x3
    return in_x1, in_x2, new_in_x3, in_y,ret

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(config._floatX)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def gen_embeddings(word_dict, dim, in_file=None,
                   init=lasagne.init.Uniform()):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
    embeddings = init((num_words, dim))
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        initialized = {}
        avg_sigma = 0
        avg_mu = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                initialized[sp[0]] = True
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
                mu = embeddings[word_dict[sp[0]]].mean()
                #print embeddings[word_dict[sp[0]]]
                sigma = np.std(embeddings[word_dict[sp[0]]])
                avg_mu += mu
                avg_sigma += sigma
        avg_sigma /= 1. * pre_trained
        avg_mu /= 1. * pre_trained
        for w in word_dict:
            if w not in initialized:
                embeddings[word_dict[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
        logging.info('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic
