import tensorflow as tf
#from tensorflow import keras
import keras as keras
import numpy as np
import re, string
import csv
from tensorflow.keras.backend import manual_variable_initialization 
import os
import constants as const

def extract_from(tokenized_fs, vocab, classes, num_sent_uppr = 36, sent_len_upper = 256):
	files = [f for f in sorted(os.listdir(tokenized_fs))]
	max_num_sentences = -1
	max_sentence_length = -1
	documents = []
	slens = []
	for f in files:
		sentences = [[vocab(c) for c in s.strip('\n')] for s in list(open(tokenized_fs + f))]
		max_num_sentences = min(max(len(sentences), max_num_sentences), num_sent_uppr)
		max_sentence_length = min(max(np.max([len(s) for s in sentences]), max_sentence_length), sent_len_upper)
		documents += [sentences]

	# USED TO DETERMINE UPPERBOUNDs ON SENTENCE LENGTH AND NUMBER OF SENTENCES FROM TRAINING DATA
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

	for i, text in enumerate(data):
		if len(text) > max_num_sentences:
			text = text[(len(text) - max_num_sentences):]   # truncate to tail
		for j, sentence in enumerate(text):
			if len(sentence) > max_sentence_length:
				sentence = sentence[(len(sentence) - max_sentence_length):]     #truncate to tail
			for k, c in enumerate(sentence):
				x[i,j,k] = c
	return x


#does the testing on the development set, returns an array with the predicted label for each piece of data 
def get_predicted_labels():
	#set up all the data. This will require calls to model_input and extract_from

	#call extract_from to get some of the required meta data before we call model.predict 
	test_data, _ , _= extract_from(const.tokenized_dev, const.vocab, const.classes)
	#convert the input to a form the model can understand 
	#NOTE: max_num_sentences and max_sentences are set to the default max for now
	max_num_sentences = 36
	max_sentence_length = 256
	x_test = model_input(test_data, max_num_sentences, max_sentence_length, const.classes)

    #load the model from memory. Note that this will be the only line that needs to change if we change models 
	mod = keras.models.load_model(const.model_to_load, custom_objects = {'tf':tf,'const':const})

	#predicted labels is a NxM NUMpy matrix:
	#	N = the number of data samples (100)
	#	M = the number of classes 
	#labels given by a probability distribution. Take the max over the columns to get the predicted label
	# for one piece of data 
	predicted_labels_probs = mod.predict(x_test)
	predicted_labels = []
	#there is 100% a better, cleaner, faster way to do this
	for i in range(0, len(x_test)):
		max_prob = 0
		max_prob_label = 0
		for j in range(0, len(const.classes)):
			if predicted_labels_probs[i,j] > max_prob:
				max_prob = predicted_labels_probs[i,j]
				max_prob_label = j
		predicted_labels.append(max_prob_label)
	return predicted_labels

def main():
	predicted_labels = get_predicted_labels()

	#export the predictions into a CSV file 
	with open(const.csv_out_file, mode='w') as pred_file:
		line_writer = csv.writer(pred_file, delimiter=',')
		file_names = [f for f in sorted(os.listdir(const.tokenized_dev))]
		for i, line in enumerate(file_names):
			#response_info[0] is the filename 
			response_info = re.split(',', line)
			for cl in const.classes:
				if const.classes[cl] == predicted_labels[i]:
					line_writer.writerow([response_info[0], cl])


if __name__ == '__main__':
    main()
		
