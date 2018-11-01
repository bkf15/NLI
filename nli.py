import re
import nltk
from nltk import bigrams
#use tokenized data. It adds a space between words and punctuation, so when parsing words we won't get a comma/period
# added to the end of a word. No conceivable advantage to using non-tokenized

#index-training.csv: column separate values of the form
# file_name   |   prompt#    |    native language    |   exam score 
#use this to get information about the file, before reading it

csv = open('TOEFL11-TRAIN/data/text/index-training.csv')

#md = array of all the mega docs
#langs contains the languages. lang[i] = the lanuage of mds[i]
langs = []
mds = []

#each line in csv corresponds to a different piece of training data, IE an essay
for line in csv:
	#splits each line into array of length 4
	# response_info[0] = file_name
	# response_info[1] = prompt
	# response_info[2] = native language
	# response_info[3] = score, although I don't think we'll even need this
	response_info = re.split(',', line)

	if response_info[2] not in langs:
		langs.append(response_info[2])

	#define path name for training file 
	response_path_name = 'TOEFL11-TRAIN/data/text/responses/tokenized/' + response_info[0]
	#let response be the file with the essay in it
	# go through the response file, add its data to MD
	with open(response_path_name) as response:
		#add essay to appropriate MD
		if len(mds) < langs.index(response_info[2]) + 1:
			mds.append(response.read().replace('\n', ' '))
		else:
			mds[langs.index(response_info[2])] += response.read().replace('\n', '')






