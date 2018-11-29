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

    # NOTE: (RE: FILEPATHS) 
    # All relevant info is marked by comment blocks with a corresponding number:
    # 1) filepaths to set directories/files for train and development data
    # 2) filepath to set model to train. Defaults to an untrained model
    #    built exactly like our turned in model.
    # 3) filepath to specify where to save best model over all epochs


    ##########################################################################
    # Start (1) ######################################################################
    ##########################################################################

    # See filepaths in constants.py

    ########################################################################
    # End (1) ####################################################################
    ########################################################################

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

    ########################################################################
    # Start (2) ####################################################################
    ########################################################################

    # set model path here
    # use 'untrained_model.hdf5' to train a model from scratch
    # can also continue the training process of any other models
    # see the NOTE: (RE UNTRAINED MODEL) in model.py
    # to see how this model was created

    model_path = 'untrained_model.hdf5'

    #########################################################################
    # End (2) #####################################################################
    #########################################################################

    # load model
    model = keras.models.load_model(model_path, custom_objects = {'tf' : tf})

    #########################################################################
    # Start (3) #####################################################################
    #########################################################################

    # modify this file path to avoid overwrites (across different validation runs) if needed

    save_path = 'newly_trained_model.hdf5'

    #########################################################################
    # End (3) #####################################################################
    #########################################################################


    # this checkpoint will save the best model seen over all epochs (on validation data)
    checkpoint = keras.callbacks.ModelCheckpoint(save_path,
        monitor = 'val_loss', save_best_only = True)
    
    batch_size = 64

    # modify this epoch number if needed, our best model took 32 epochs before gradually starting to overfit
    # checkpoint callback will save/overwrite model after each epoch where there is improvement on validation loss
    # 35 epochs is about 6 hrs on  iMac, 3.3 GHz i7, 16 GB RAM 1867 MHz DDR3, Intel Iris Pro Graphics 6200 1536 MB
    num_epochs = 35
    
    model.fit(x_train, y_train,
          batch_size = batch_size, epochs = num_epochs,
          validation_data=(x_val, y_val), callbacks = [checkpoint])


if __name__ == '__main__':
    main()