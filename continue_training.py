import tensorflow as tf
import keras
import numpy as np
import random as rd
import string
import os
import constants as const
from model import extract_from
from model import model_input

# Q: What does this file do?
# A: continues training (or starts training) models built with model.py
##########################################################################

def main():

    # extract character indeces per sentence per document for training and development
    train, _, _ = extract_from(const.tokenized_train, const.train_labels, const.vocab, const.classes)
    dev, _, _ = extract_from(const.tokenized_dev, const.dev_labels, const.vocab, const.classes)

    # these params are fixed now b/c model is finalized 
    # model expects the same input dimensions
    max_num_sentences = 36
    max_sentence_length = 256

    # build training/development input and labeled output
    x_train, y_train = model_input(train, max_num_sentences, max_sentence_length, const.classes)
    x_val, y_val = model_input(dev, max_num_sentences, max_sentence_length, const.classes)

    # load model
    model = keras.models.load_model(const.model_to_train, custom_objects = {'tf' : tf})

    # this checkpoint will save the best model seen over all epochs (on validation data)
    checkpoint = keras.callbacks.ModelCheckpoint(const.save_path_for_model,
        monitor = 'val_loss', save_best_only = True)
    
    batch_size = 64

    # NOTE: NUMBER OF EPOCHS
    # modify this epoch number if needed, our best model took 32 epochs before gradually starting to overfit
    # checkpoint callback will save/overwrite model after each epoch where there is improvement on validation loss
    # 35 epochs is about 6 hrs on  iMac, 3.3 GHz i7, 16 GB RAM 1867 MHz DDR3, Intel Iris Pro Graphics 6200 1536 MB
    num_epochs = 35
    
    model.fit(x_train, y_train,
          batch_size = batch_size, epochs = num_epochs,
          validation_data=(x_val, y_val), callbacks = [checkpoint])

if __name__ == '__main__':
    main()