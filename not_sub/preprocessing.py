import tensorflow as tf
import numpy as np
import constants as const
import autocorrect as ac
import random as rd
import os


def extract(tokenized_fs, label_fs, perc = 97):
    labels = [l.split(',') for l in list(open(label_fs))]
    files = [f for f, _, _, _ in labels]
    labels = [lb for _, _, lb, _ in labels]
    doc_lens = []
    sent_lens = []
    word_lens = []
    #char level documents
    documents_cl = []
    #word level documents
    documents_wl = []
    word_vocab = dict()

    for f in files:

        # Char level document
        # a document is a collection of sentences
        # a sentence is a collection of words
        # a word is a collection of characters
        sentences_cl = [[[c for c in w]
            for w in s.strip('\n').strip('.').split(' ') if w != '']
            for s in list(open(tokenized_fs + f))]
        
        # Word level document
        # a document is a collection of sentences
        # a sentence is a collection of words
        sentences_wl = [[w
            for w in s.strip('\n').strip('.').split(' ') if w != '']
            for s in list(open(tokenized_fs + f))]

        # identify word vocab
        for s in sentences_cl:
            for w in s:
                w = ''.join(w)
                if w not in word_vocab:
                    word_vocab[w] = 0
                word_vocab[w] += 1

        # add sentence to word level and chrar level documents
        documents_wl += [sentences_wl]
        documents_cl += [sentences_cl]

        # record lengths to determine upperbounds
        doc_lens += [len(sentences_cl)]
        sent_lens += [len(s) for s in sentences_cl]
        word_lens += [len(w) for s in sentences_cl for w in s]

    # Info. on distribution of percentiles to pick upperbounds from
    # training data
    
    # # output lenghts that account for chosen percentile
    # print('Document Length ' + str(perc) +
    #     ' Percentile:')
    # print('\t' + str(np.percentile(doc_lens, perc)))
    # print('Sentence Length ' + str(perc) +
    #     ' Percentile:')
    # print('\t' + str(np.percentile(sent_lens, perc)))
    # print('Word Length ' + str(perc) +
    #     ' Percentile:')
    # print('\t' + str(np.percentile(word_lens, perc)))
    # print('Word Count ' + str(perc) + 
    #     ' Percentile:')
    # print('\t' + str(np.percentile(
    #     list(word_vocab.values()), perc)))
    # # output word counts
    # print('Word Vocab: ')
    # for k, v in word_vocab.items():
    #     print(k + ' had ' + str(v) + ' appearances.')
    # # output word vocab length
    # print('Number of seen words:')
    # print('\t' +str(len(word_vocab)))

    return documents_cl, documents_wl, labels

def doc_model_batch_gen(documents, labels, batch_size, vec):

    seen = len(documents) + 1

    while True:

        if seen > len(documents):
            group = list(zip(documents, labels))
            rd.shuffle(group)
            documents, labels = zip(*group)
            seen = 0

        # data representation x represents documents, y labels
        x_vec = np.zeros((batch_size, const.doc_len_ub, const.sent_len_ub, 
            const.word_vec_size), dtype = np.float)
        x_char = np.zeros((batch_size, const.doc_len_ub, const.sent_len_ub, 
            const.word_len_ub), dtype = np.int64)
        y = np.zeros((batch_size, len(const.classes)))

        # if batch sizes don't fit in steps per epoch
        batch_stop = batch_size \
            if seen + batch_size <= len(documents) \
            else len(documents) - seen

        for i in range(batch_stop):
            document = documents[seen + i]
            document = document[(len(document) - const.doc_len_ub):]

            for j, sentence in enumerate(document):
                # truncate to tail
                sentence = sentence[(len(sentence) - const.sent_len_ub):]

                for k, word in enumerate(sentence):
                    # x_vec[i, j, k, l] is ...
                    # the lth component of ...
                    # the kth word-vector of ...
                    # the jth sentence of ...
                    # the ith document.
        
                    try: 
                        # if word is in dictionary
                        x_vec[i, j, k, :] = vec[word]
                    except KeyError:
                        try:
                            # try again with spell-check on
                            x_vec[i, j, k, :] = vec[ac.spell(word)]
                        except KeyError:
                            # can't determine, so random
                            x_vec[i, j, k, :] = np.random.uniform(-1, 1, 
                                    const.word_vec_size)

                    word = word[(len(word) - const.word_len_ub):]
            
                    for l, char in enumerate(word):
                        # x_char[i, j, k, l] is ...
                        # the lth character of ...
                        # the kth word of ...
                        # the jth sentence of ...
                        # the ith document.
                        x_char[i, j, k, l] = const.vocab(char)

            # y[i, j] is 1 if the ith document is labeled as language j
            # and 0 otherwise
            y[i, :] = np.array([1 if cl == labels[seen + i] else 0 
                for cl in const.classes.keys()])

        seen += batch_size
        yield [x_vec, x_char], y

def one_hot_(x):
    return tf.to_float(tf.one_hot(x, const.vocab_len, on_value = 1, off_value = 0))

def one_hot_out_(in_shape):
    return (in_shape[0], in_shape[1], const.vocab_len)
    

