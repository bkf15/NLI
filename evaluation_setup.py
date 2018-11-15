import tensorflow as tf
from tensorflow import keras
import numpy as np
import re, string
import csv
from tensorflow.keras.backend import manual_variable_initialization 
import os
os.environ['PYTHONHASHSEED'] = '0'

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


#does the testing on the development set, returns an array with the predicted label for each piece of data 
def get_predicted_labels():
	#set up all the data. This will require calls to model_input and extract_from
	#initialize the vocabulary to be all the ascii letters, plus digits, punc, etc...
	vocabulary = set((list(string.ascii_letters) 
        +list(string.digits)
        +list(string.punctuation)
        +list(' ')     # just space
        +list(string.digits)))
	vocabulary = {c : i + 1 for i, c in enumerate(vocabulary)}
	vocab = lambda c: vocabulary[c] if c in vocabulary else len(vocabulary) + 1

    #the model will just output a number, so we need to tie this identifier to the actual input classes (the lang)
	classes = ['TUR', 'ARA', 'HIN', 'KOR', 'ITA', 'ZHO', 'TEL', 'SPA', 'JPN', 'FRA', 'DEU']
	classes = {cl : i for i, cl in enumerate(classes)}
	_classes = {i : cl for cl, i in classes.items()}

    #save the file path for the dev data as a variable
	tokenized_dev = 'TOEFL11-DEV/data/text/responses/tokenized/'
	dev_labels = 'TOEFL11-DEV/data/text/index-dev.csv'

	#call extract_from to get some of the required meta data before we call model.predict 
	test, max_num_sentences, max_sentence_length = extract_from(tokenized_dev, dev_labels, vocab, classes)
	#convert the input to a form the model can understand 
	#NOTE: max_num_sentences and max_sentences are set to the default max for now
	x_test, y_test = model_input(test, 36, 256, classes)

    #load the model from memory. Note that this will be the only line that needs to change if we change models 
	mod = keras.models.load_model('sent_test_model.h5')
	#mod.load_weights('sent_test_model.h5')

	#predicted labels is a NxM NUMpy matrix:
	#	N = the number of data samples (100)
	#	M = the number of classes 
	#labels given by a probability distribution. Take the max over the columns to get the predicted label
	# for one piece of data 
	#print(mod.get_weights()) 
	predicted_labels_probs = mod.predict(x_test)
	#predicted_labels_probs = mod.predict_classes(x_test)
	predicted_labels = []
	#there is 100% a better, cleaner, faster way to do this
	for i in range(0, 1100):
		max_prob = 0
		max_prob_label = 0
		for j in range(0, 11):
			if predicted_labels_probs[i,j] > max_prob:
				max_prob = predicted_labels_probs[i,j]
				max_prob_label = j
		predicted_labels.append(max_prob_label)
	#print(predicted_labels_probs[0])
	#print(predicted_labels[0])
	return predicted_labels

def main():
	classes = ['TUR', 'ARA', 'HIN', 'KOR', 'ITA', 'ZHO', 'TEL', 'SPA', 'JPN', 'FRA', 'DEU']
	classes = {cl : i for i, cl in enumerate(classes)}
	predicted_labels = get_predicted_labels()
	correct_predictions = 0
	with open('TOEFL11-DEV/data/text/index-dev.csv') as f:
		for i, line in enumerate(f):
			response_info = re.split(',', line)
			real_label = classes[response_info[2]]
			if real_label == predicted_labels[i]:
				correct_predictions += 1
	#print('Correct predictions: ' + str(correct_predictions))

	#export the predictions into a CSV file 
	with open('dev_predictions.csv', mode='w') as pred_file:
		line_writer = csv.writer(pred_file, delimiter=',')
		file_names = open('TOEFL11-DEV/data/text/index-dev.csv')
		for i, line in enumerate(file_names):
			#response_info[0] is the filename 
			response_info = re.split(',', line)
			for cl in classes:
				if classes[cl] == predicted_labels[i]:
					line_writer.writerow([response_info[0], cl])


if __name__ == '__main__':
    main()
		
