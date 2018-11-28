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
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
        const.google_pretrained_word2vec, binary = True)

    doc_input = keras.Input(shape = (const.doc_len_ub, const.sent_len_ub, const.word_vec_size, ), dtype = 'float')
    sent_input = keras.Input(shape = (const.sent_len_ub, const.word_vec_size, ), dtype = 'float')

    # Conv and lstm for word2vec rep of sentence
    filter_sizes = [2, 4, 5]
    num_filters = [100, 100, 100]
    conv_layers = list(zip(filter_sizes, num_filters))
    dropout = .5
    reg = keras.constraints.MaxNorm(3)
    act = 'relu'
    hidden_units = 128
    lstm_dropout = 0.15
    rec_dropout = 0.15

    conv_out = []

    for filter_size, num in conv_layers:
        conv = keras.layers.Conv1D(filters = num, kernel_size = filter_size,
            activation = act, kernel_constraint = reg)(sent_input)
        #conv = keras.layers.Dropout(rate = dropout)(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.GlobalMaxPool1D()(conv)
        conv_out += [conv]
    
    sent_encoding = keras.layers.Concatenate()(conv_out)
    sent_model = keras.Model(inputs = sent_input, outputs = sent_encoding)

    keras.utils.plot_model(sent_model, to_file='sent_model.svg')
    doc_stack = keras.layers.TimeDistributed(sent_model)(doc_input)
    doc_rep = keras.layers.LSTM(hidden_units, dropout = lstm_dropout, 
        recurrent_dropout = rec_dropout)(doc_stack)

    output = keras.layers.Dense(128, activation='relu')(doc_rep)
    output = keras.layers.Dropout(0.3)(output)
    output = keras.layers.Dense(len(const.classes), activation='softmax')(output)
    model = keras.Model(inputs = doc_input, outputs = output)
    keras.utils.plot_model(model, to_file='model_two.svg')
    
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    batch_size = 64
    num_epochs = 25
    patience = 5
    min_delta = .1

    stop_criterion = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
        patience = patience, min_delta = min_delta)
    checkpoint = keras.callbacks.ModelCheckpoint('model_two.{epoch:02d}.hdf5',
        monitor = 'val_loss', save_best_only = True)
    train_bgen = pp.doc_model_batch_gen(train_docs, train_labels, 
        batch_size, w2v_model)

    val_bgen = pp.doc_model_batch_gen(dev_docs, dev_labels, 
        batch_size, w2v_model)

    model.fit_generator(train_bgen, steps_per_epoch = len(train_docs) // batch_size, 
        validation_data = val_bgen, validation_steps = len(dev_docs) // batch_size,
        epochs = num_epochs, callbacks = [stop_criterion, checkpoint])

    # backup save last epoch just in case
    model.save('model_two.h5')

if __name__ == '__main__':
    main()