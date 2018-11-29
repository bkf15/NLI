from tensorflow import keras
import constants as const
import preprocessing as pp
import tensorflow as tf
import numpy as np
import gensim

def main():

    # extract documents and labels
    _, train_docs, train_labels = pp.extract(const.tokenized_train, 
        const.train_labels)
    _, dev_docs, dev_labels = pp.extract(const.tokenized_dev,
        const.dev_labels)

    # Note this model is case-sensitive !
    vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        const.google_pretrained_word2vec, binary = True)

    doc_input = keras.Input(shape = (const.doc_len_ub, const.sent_len_ub, const.word_vec_size, ), dtype = 'float')
    sent_input = keras.Input(shape = (const.sent_len_ub, const.word_vec_size, ), dtype = 'float')

    filter_sizes = [2, 4, 5]
    num_filters = [100, 100, 100]
    conv_layers = list(zip(filter_sizes, num_filters))
    dropout = .5
    reg = keras.constraints.MaxNorm(3)
    act = 'relu'

    conv_out = []

    for filter_size, num in conv_layers:
        conv = keras.layers.Conv1D(filters = num, kernel_size = filter_size,
            activation = act, kernel_constraint = reg)(sent_input)
        # conv = keras.layers.Dropout(rate = dropout)(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.GlobalMaxPool1D()(conv)
        conv_out += [conv]
    
    sent_rep = keras.layers.Concatenate()(conv_out)
    sent_model = keras.Model(inputs = sent_input, outputs = sent_rep)
    keras.utils.plot_model(sent_model, to_file='sent_model_three.svg')

    # run same above process on the document stack

    doc_stack = keras.layers.TimeDistributed(sent_model)(doc_input)
    
    filter_sizes = [2, 4, 5]
    num_filters = [100, 100, 100]
    conv_layers = list(zip(filter_sizes, num_filters))
    dropout = .5
    reg = keras.constraints.MaxNorm(3)
    act = 'relu'

    conv_out = []

    for filter_size, num in conv_layers:
        conv = keras.layers.Conv1D(filters = num, kernel_size = filter_size,
            activation = act, kernel_constraint = reg)(doc_stack)
        # conv = keras.layers.Dropout(rate = dropout)(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.GlobalMaxPool1D()(conv)
        conv_out += [conv]
    
    doc_rep = keras.layers.Concatenate()(conv_out)

    output = keras.layers.Dropout(rate = 0.3)(doc_rep)
    output = keras.layers.Dense(128, activation = 'relu',
        kernel_constraint = reg)(output)
    output = keras.layers.Dropout(rate = 0.3)(output)
    output = keras.layers.Dense(len(const.classes), activation = 'softmax',
        kernel_constraint = reg)(output)

    model = keras.Model(inputs = doc_input, outputs = output)
    keras.utils.plot_model(model, to_file='model_three.svg')

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    batch_size = 64
    num_epochs = 25
    patience = 5
    min_delta = .1

    stop_criterion = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
        patience = patience, min_delta = min_delta)
    checkpoint = keras.callbacks.ModelCheckpoint('model_three.{epoch:02d}.hdf5',
        monitor = 'val_loss', save_best_only = True)
    train_bgen = pp.doc_model_batch_gen_2(train_docs, train_labels, 
        batch_size, vec_model)

    val_bgen = pp.doc_model_batch_gen_2(dev_docs, dev_labels, 
        batch_size, vec_model)

    model.fit_generator(train_bgen, steps_per_epoch = len(train_docs) // batch_size, 
        validation_data = val_bgen, validation_steps = len(dev_docs) // batch_size,
        epochs = num_epochs, callbacks = [stop_criterion])
    
    model.save('model_three.h5')

if __name__ == '__main__':
    main()