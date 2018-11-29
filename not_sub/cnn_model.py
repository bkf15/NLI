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

    vec_doc_input = keras.Input(shape = (const.doc_len_ub, const.sent_len_ub, const.word_vec_size, ), dtype = 'float')
    char_doc_input = keras.Input(shape = (const.doc_len_ub, const.sent_len_ub, const.word_len_ub, ), dtype = 'int64')
    vec_sent_input = keras.Input(shape = (const.sent_len_ub, const.word_vec_size, ), dtype = 'float')
    char_sent_input = keras.Input(shape = (const.sent_len_ub, const.word_len_ub, ), dtype = 'int64')
    char_input = keras.layers.Flatten()(char_sent_input)

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
            activation = act, kernel_constraint = reg)(vec_sent_input)
        #conv = keras.layers.Dropout(rate = dropout)(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.GlobalMaxPool1D()(conv)
        conv_out += [conv]
    
    vec_sent_encoding = keras.layers.Concatenate()(conv_out)
    vec_model = keras.Model(inputs = vec_sent_input, outputs = vec_sent_encoding)
    keras.utils.plot_model(vec_model, to_file='vec_model.svg')
    vec_doc_stack = keras.layers.TimeDistributed(vec_model)(vec_doc_input)
    vec_lstm = keras.layers.LSTM(hidden_units, dropout = lstm_dropout, 
        recurrent_dropout = rec_dropout)(vec_doc_stack)
    
    # Conv and lstm for char rep of sentence
    filter_sizes = [5, 3, 3]
    num_filters = [196, 196, 256]
    conv_layers = list(zip(filter_sizes, num_filters))
    dropout = .1
    reg = keras.constraints.MaxNorm(3)
    act = 'relu'
    hidden_units = 128
    lstm_dropout = 0.15
    rec_dropout = 0.15

    conv = keras.layers.Lambda(pp.one_hot_, output_shape = pp.one_hot_out_)(char_input)

    for filter_size, num in conv_layers:
        conv = keras.layers.Conv1D(filters = num, kernel_size = filter_size,
            activation = act, kernel_constraint = reg)(conv)
        #conv = keras.layers.Dropout(rate = dropout)(conv)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.MaxPool1D(pool_size = 2)(conv)

    char_sent_encoding = keras.layers.Bidirectional(
        keras.layers.LSTM(hidden_units, dropout = lstm_dropout, 
        recurrent_dropout = rec_dropout))(conv)
    # char_sent_encoding = keras.layers.Dropout(rate = 0.3)(char_sent_encoding)

    char_model = keras.Model(inputs = char_sent_input, outputs = char_sent_encoding)
    keras.utils.plot_model(char_model, to_file='char_model.svg')
    char_doc_stack = keras.layers.TimeDistributed(char_model)(char_doc_input)
    char_lstm = keras.layers.LSTM(hidden_units, dropout = lstm_dropout, 
        recurrent_dropout = rec_dropout)(char_doc_stack)
    
    doc_rep = keras.layers.Concatenate()([vec_lstm, char_lstm])
    # test = keras.Model(inputs = [vec_doc_input, char_doc_input], outputs = doc_rep)
    # # keras.utils.plot_model(test, to_file='model.svg')
    # # test.summary()
    # # print(test.output_shape)
    # # exit()

    output = keras.layers.Dense(128, activation='relu')(doc_rep)
    output = keras.layers.Dropout(0.3)(output)
    output = keras.layers.Dense(len(const.classes), activation='softmax')(output)
    model = keras.Model(inputs = [vec_doc_input, char_doc_input], outputs = output)
    keras.utils.plot_model(model, to_file='model.svg')
    
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    batch_size = 32
    num_epochs = 25
    patience = 5
    min_delta = .1

    stop_criterion = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
        patience = patience, min_delta = min_delta)
    checkpoint = keras.callbacks.ModelCheckpoint('dual_cnn_model.{epoch:02d}.hdf5',
        monitor = 'val_loss', save_best_only = True)
    train_bgen = pp.doc_model_batch_gen(train_docs, train_labels, 
        batch_size, w2v_model)

    val_bgen = pp.doc_model_batch_gen(dev_docs, dev_labels, 
        batch_size, w2v_model)

    model.fit_generator(train_bgen, steps_per_epoch = len(train_docs) // batch_size, 
        validation_data = val_bgen, validation_steps = len(dev_docs) // batch_size,
        epochs = num_epochs, callbacks = [stop_criterion, checkpoint])

    # backup save last epoch just in case
    model.save('dual_cnn_model.h5')

if __name__ == '__main__':
    main()