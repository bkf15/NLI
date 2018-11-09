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
	csv.close()
	return md

#writes to a file called unique_words.txt
# writes out all words that occur in the corpus. Words that appear many times in the corpus only
# appear once in the file. 
def getUniqueWords():
	csv = open('TOEFL11-TRAIN/data/text/index-training.csv')
	words = []
	for line_in_list in csv:
		response_info = re.split(',', line_in_list)
		response_path_name = 'TOEFL11-TRAIN/data/text/responses/tokenized/' + response_info[0]
		with open(response_path_name) as response:
			for line_in_doc in response:
				word_list = re.sub("[^\w]", " ",  line_in_doc).split()
				for word in word_list:
					if word not in words:
						words.append(word)
	f = open('unique_words.txt', 'w')
	for word in words:
		f.write(word + '\n')
	f.close

#writes out to a file called unique_chars.txt that will contain all the chars that appear in corpus
def getUniqueChars():
	csv = open('TOEFL11-TRAIN/data/text/index-training.csv')
	chars = []
	for line_in_list in csv:
		response_info = re.split(',', line_in_list)
		response_path_name = 'TOEFL11-TRAIN/data/text/responses/tokenized/' + response_info[0]
		with open(response_path_name) as response:
			for line_in_doc in response:
				for char in line_in_doc:
					if char not in chars:
						chars.append(char)
	f = open('unique_chars.txt', 'w')
	for char in chars:
		f.write(char + '\n')
	f.close()

getUniqueWords()

#md = setupTrainingData()
#print(md[3][0])

#want: words on different lines, characters separated by spaces
# document indexed by words (doc[i] = word_i)
#   periods as their own words 
# mega document per language, on each line specify:
#    documentindex | wordindex | actualword | POS 
# find max sentence length !!!!
# 


#write a random baseline, randomly select NL



