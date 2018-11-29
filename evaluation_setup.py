import tensorflow as tf
#from tensorflow import keras
import keras as keras
import numpy as np
import re, string
import csv
from tensorflow.keras.backend import manual_variable_initialization 
import os
import constants as const
from model import extract_from, model_input
print(keras.__version__)

#does the testing on the development set, returns an array with the predicted label for each piece of data 
def get_predicted_labels():
	#set up all the data. This will require calls to model_input and extract_from

	#call extract_from to get some of the required meta data before we call model.predict 
	test_data, _ , _= extract_from(const.tokenized_dev, const.dev_labels, const.vocab, const.classes)
	#convert the input to a form the model can understand 
	#NOTE: max_num_sentences and max_sentences are set to the default max for now
	max_num_sentences = 36
	max_sentence_length = 256
	x_test, y_test = model_input(test_data, max_num_sentences, max_sentence_length, const.classes)

    #load the model from memory. Note that this will be the only line that needs to change if we change models 
	mod = keras.models.load_model('trained_model.hdf5', custom_objects = {'tf':tf,'const':const})
	#mod.load_weights('sent_test_model.h5')

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
	correct_predictions = 0
	with open(const.dev_labels) as f:
		for i, line in enumerate(f):
			response_info = re.split(',', line)
			real_label = const.classes[response_info[2]]
			if real_label == predicted_labels[i]:
				correct_predictions += 1

	#export the predictions into a CSV file 
	with open('predictions.csv', mode='w') as pred_file:
		line_writer = csv.writer(pred_file, delimiter=',')
		file_names = open(const.dev_labels)
		for i, line in enumerate(file_names):
			#response_info[0] is the filename 
			response_info = re.split(',', line)
			for cl in const.classes:
				if const.classes[cl] == predicted_labels[i]:
					line_writer.writerow([response_info[0], cl])


if __name__ == '__main__':
    main()
		
