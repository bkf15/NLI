import string


# File paths
# --------------------------------------------------------------
#######################################################################


# set tokenized train to be a directory of tokenized files to train on
tokenized_train = 'TOEFL11-TRAIN/data/text/responses/tokenized/'
# set index for training files above
train_labels = 'TOEFL11-TRAIN/data/text/index-training.csv'

# set tokenized dev to be a directory of tokenized files for development
tokenized_dev = 'TOEFL11-DEV/data/text/responses/tokenized/'
# set index for training files above
dev_labels = 'TOEFL11-DEV/data/text/index-dev.csv'

# model being loaded by 'evaluation_setup' for predictions
# defaults to our original trained model (32 epochs ~ 6 hours)
model_to_load = 'trained_model.hdf5'

# where to save best model over all epochs, for 'continue_training.py'
save_path_for_model = 'newly_trained.hdf5'

# model to load for training in 'continue_training.py'
# defaults to untrained model built exactly as original
model_to_train = 'untrained_model.hdf5'

# file name for the .csv predictions output
csv_out_file = 'predictions.csv'

#######################################################################
# -------------------------------------------------------------



# Char Vocabulary
# ------------------------------------------------------

_vocabulary = set((list(string.ascii_letters) 
        + list(string.digits)
        + list(string.punctuation))
		+ list(' '))

vocabulary = {c : i + 1 for i, c in enumerate(_vocabulary)}
vocab = lambda c: vocabulary[c] if c in vocabulary else len(vocabulary) + 1

vocab_len = len(vocabulary) + 2



# Classes
# --------------------------------------------------------------
_classes = ['TUR', 'ARA', 'HIN', 'KOR', 'ITA', 'ZHO', 'TEL', 'SPA', 'JPN', 'FRA', 'DEU']
classes = {cl : i for i, cl in enumerate(_classes)}
