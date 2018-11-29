import tensorflow as tf
import keras
import numpy as np
import random as rd
import string
import os
import constants as const
from model import extract_from
from model import model_input

def main():


    # CHANGE FILE PATHS BELOW
    ######################################################################
    # set tokenized train to be a directory of tokenized files to train on
    tokenized_train = 'TOEFL11-TRAIN/data/text/responses/tokenized/'
    # set index for training files above
    train_labels = 'TOEFL11-TRAIN/data/text/index-training.csv'

    # set tokenized dev to be a directory of tokenized files for development
    tokenized_dev = 'TOEFL11-DEV/data/text/responses/tokenized/'
    # set index for training files above
    dev_labels = 'TOEFL11-DEV/data/text/index-dev.csv'
    ######################################################################

    # extract character indeces per sentence per document for training and development
    train, _, _ = extract_from(tokenized_train, train_labels, const.vocab, const.classes)
    dev, _, _ = extract_from(tokenized_dev, dev_labels, const.vocab, const.classes)

    # these params are fixed now b/c model is finalized 
    # model expects the same input dimensions
    max_num_sentences = 36
    max_sentence_length = 256

    # build training/development input and labeled output
    x_train, y_train = model_input(train, max_num_sentences, max_sentence_length, const.classes)
    x_val, y_val = model_input(dev, max_num_sentences, max_sentence_length, const.classes)

    # set model path here
    # use 'untrained_model.hdf5' to train a model from scratch
    # can also continue training any models terminated to early
    model_path = 'trained_model.hdf5'

    model = keras.models.load_model(model_path, custom_objects = {'tf' : tf})

    # modify this file bath to avoid overwrites if needed
    # this checkpoint will save the best model seen over all epochs (on validation data)
    checkpoint = keras.callbacks.ModelCheckpoint('model.{epoch:02d}.hdf5',
        monitor = 'val_loss', save_best_only = True)
    
    batch_size = 64

    #modify this epoch number if needed, our best model took 32 epochs before gradually starting to overfit 
    num_epochs = 100
    
    model.fit(x_train, y_train,
          batch_size = batch_size, epochs = num_epochs,
          validation_data=(x_val, y_val), callbacks = [checkpoint])


if __name__ == '__main__':
    main()