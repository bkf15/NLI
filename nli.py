import re
import nltk
from nltk import bigrams

#returns an array of tuples md[i] = (li, di) where di is all of the documents of langauge li 
def setupTrainingData():
	csv = open('TOEFL11-TRAIN/data/text/index-training.csv')
	#mds = array of all the mega docs, IE mds[i] = all the documents written by people whos native language is langs[i]
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

		#if the language hasn't been encountered before, add it to langs
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

	#now, perform modeling on the mega documents 
	md = []
	for i in range(0, len(langs)):
		md.append((langs[i], mds[i]))
	return md



md = setupTrainingData()








