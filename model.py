import tensorflow as tf
import keras
import numpy as np
import random as rd
import string
import os

# What does this file do?
# Builds, trains, and saves models
# saves untrained model as well (currently commented out)
##########################################################################

def extract_from(tokenized_fs, label_fs, vocab, classes, num_sent_uppr = 36, sent_len_upper = 256):
    labels = [l.split(',') for l in list(open(label_fs))]
    labels = {f : lb for f , _ , lb, _ in labels}
    max_num_sentences = -1
    max_sentence_length = -1
    documents = []
    slens = []
    for f in labels:
        sentences = [[vocab(c) for c in s.strip('\n')] for s in list(open(tokenized_fs + f))]
        max_num_sentences = min(max(len(sentences), max_num_sentences), num_sent_uppr)
        max_sentence_length = min(max(np.max([len(s) for s in sentences]), max_sentence_length), sent_len_upper)
        documents += [[sentences, classes[labels[f]]]]

    # USED TO DETERMINE UPPERBOUND ON SENTENCE LENGTH AND NUMBER OF SENTENCES FROM TRAINING DATA
    ############################################################################################
        # slens += [len(s) for s in sentences]
    #     slens += [len(sentences)]
    # print(np.mean(slens))
    # print(np.percentile(slens, 99))
    # print(max_num_sentences)
    # print(max_sentence_length)
    # exit()
    ############################################################################################

    return documents, max_num_sentences, max_sentence_length

def model_input(data, max_num_sentences, max_sentence_length, classes):
    x = np.zeros((len(data), max_num_sentences, max_sentence_length), dtype = np.int64)
    y = np.zeros((len(data), len(classes)))

    for i, example in enumerate(data):
        text = example[0]
        label = example[1]
        if len(text) > max_num_sentences:
            text = text[(len(text) - max_num_sentences):]   # truncate to tail
        for j, sentence in enumerate(text):
            if len(sentence) > max_sentence_length:
                sentence = sentence[(len(sentence) - max_sentence_length):]     #truncate to tail
            for k, c in enumerate(sentence):
                x[i,j,k] = c
        y[i, :] = np.array([1 if int(cl) == int(label) else 0 for cl in  range(len(classes))])
    return x, y

def main():

    # DEFINE MODEL CONSTANTS
    # All of these params are the same as in constants
    # we did not want to modify the original script used to build the model
    ###################################################################################

    vocabulary = set((list(string.ascii_letters) 
        + list(string.digits)
        + list(string.punctuation)
        + list(' ')     # just space
        +list(string.digits)))
    vocabulary = {c : i + 1 for i, c in enumerate(vocabulary)}
    vocab = lambda c: vocabulary[c] if c in vocabulary else len(vocabulary) + 1

    classes = ['TUR', 'ARA', 'HIN', 'KOR', 'ITA', 'ZHO', 'TEL', 'SPA', 'JPN', 'FRA', 'DEU']
    classes = {cl : i for i, cl in enumerate(classes)}
    _classes = {i : cl for cl, i in classes.items()}

    tokenized_train = 'TOEFL11-TRAIN/data/text/responses/tokenized/'
    train_labels = 'TOEFL11-TRAIN/data/text/index-training.csv'
    tokenized_dev = 'TOEFL11-DEV/data/text/responses/tokenized/'
    dev_labels = 'TOEFL11-DEV/data/text/index-dev.csv'

    ####################################################################################

    # extract character indeces per sentence per document for training and development
    train, max_num_sentences, max_sentence_length = extract_from(tokenized_train, train_labels, vocab, classes)
    dev, _, _ = extract_from(tokenized_dev, dev_labels, vocab, classes)

    # build training/development input and labeled output
    x_train, y_train = model_input(train, max_num_sentences, max_sentence_length, classes)
    x_val, y_val = model_input(dev, max_num_sentences, max_sentence_length, classes)

    # Define model input
    ####################################################################################

    # document input has rows as sentences as columns as character indeces in sentence
    doc_input = keras.Input(shape = (max_num_sentences, max_sentence_length, ), dtype = 'int64')

    # each row/sentence is distributed to a sentnence input
    # this is for the sentence model within the larger document model 
    sent_intput = keras.Input(shape = (max_sentence_length, ), dtype = 'int64')

    ####################################################################################

    # Define model
    ####################################################################################

    # Characters are first embedded (keras uses a look-up table embedding)
    char_embedding = keras.layers.Embedding(len(vocabulary) + 2, 16, input_length = max_sentence_length)(sent_intput)
    

    # Define conv layers parmaeters
    filter_sizes = [2, 4, 5]
    num_filters = [100, 100, 100]
    conv_layers = list(zip(filter_sizes, num_filters))
    dropout = .1
    reg = None #keras.constraints.MaxNorm(3)
    act = 'relu'

    conv_out = []

    # Define conv arch.
    # Conv architecture taken from Yoon Kim "Convolutional Neural Networks for Sentence Classification"
    # Each conv layer creates a representation of the sentence, 1d over temporal dimension
    # Global max pooling over time
    for filter_size, num in conv_layers:
        conv = keras.layers.Conv1D(filters = num, kernel_size = filter_size,
            activation = act, kernel_constraint = reg)(char_embedding)
        conv = keras.layers.Dropout(rate = dropout)(conv)
        conv = keras.layers.GlobalMaxPool1D()(conv)
        conv_out += [conv]

    # concatnate the outputs to form a final representation of the sentence
    sent_encoding = keras.layers.Concatenate()(conv_out)
    # define sentence model within larger document model
    sent_encoder = keras.Model(inputs = sent_intput, outputs = sent_encoding)

    # use sentence model to get representation of each sentence in a document
    encoded = keras.layers.TimeDistributed(sent_encoder)(doc_input)

    # bidirectional lstm over whole document
    lstm_doc = keras.layers.Bidirectional(
        keras.layers.LSTM(128, dropout=0.15, recurrent_dropout=0.15))(encoded)

    # fully connected (dense) network with softmax output for classification
    output = keras.layers.Dropout(0.3)(lstm_doc)
    output = keras.layers.Dense(128, activation='relu')(output)
    output = keras.layers.Dropout(0.3)(output)
    output = keras.layers.Dense(len(classes), activation='softmax')(output)
    
    model = keras.Model(inputs = doc_input, outputs = output)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    # USED TO SAVE UNTRAINED MODEL FOR CROSS VAL
    # model.save('untrained_model_two.hdf5')
    # exit()

    batch_size = 64
    num_epochs = 100
    patience = 10
    min_delta = .1

    # only save best model for validation loss (form of early stopping to prevent overfitting)
    # this way we can kill training when model seems to be overfitting and we have saved 
    # the best model (on validation data)
    checkpoint = keras.callbacks.ModelCheckpoint('sent_model_three_r.{epoch:02d}.hdf5',
        monitor = 'val_loss', save_best_only = True)

    model.fit(x_train, y_train,
          batch_size = batch_size, epochs = num_epochs,
          validation_data=(x_val, y_val), callbacks = [checkpoint])

    model.save('sent_model_last_three_r.h5')

if __name__ == '__main__':
    main()
    




