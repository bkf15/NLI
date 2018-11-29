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
