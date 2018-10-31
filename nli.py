import re
import nltk
from nltk import bigrams
#use tokenized data. It adds a space between words and punctuation, so when parsing words we won't get a comma/period
# added to the end of a word. No conceivable advantage to using non-tokenized

#index-training.csv: column separate values of the form
# file_name   |   prompt#    |    native language    |   exam score 
#use this to get information about the file, before reading it

csv = open('TOEFL11-TRAIN/data/text/index-training.csv')

#each line in csv corresponds to a different piece of training data, IE an essay
for line in csv:
	#splits each line into array of length 4
	# file[0] = file_name
	# file[1] = prompt
	# file[2] = native language
	# file[3] = score, although I don't think we'll even need this
	response_info = re.split(',', line)
	#define path name for training file 
	response_path_name = 'TOEFL11-TRAIN/data/text/responses/tokenized/' + response_info[0]
	#let response be the file with the essay in it
	# go through the response file, add its data to our models
	with open(response_path_name) as response:
		#get word n-gram data?
		#get character n-gram data?
		#get POS data?
		#add data to our models

		#note that because we are using the tokenized responses, we do not need to worry about separating punctuation 
		# from the words. Will probably just choose to ignore punctuation, but we'll see 

		for l in response:
			file_bigram = list(bigrams(l.split()))
			print(file_bigram)
		break




