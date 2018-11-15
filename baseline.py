import re
import random
#function that literally just randomly assigns an NL to each document
# used as our baseline 

def naiveBaseline():
	langs = ['DEU', 'TUR', 'ZHO', 'TEL', 'ARA', 'SPA', 'HIN', 'JPN', 'KOR', 'ITA', 'FRA']
	correct_assignments = [0,0,0,0,0,0,0,0,0,0,0]
	false_positives = [0,0,0,0,0,0,0,0,0,0,0]
	num_responses_per_language = [0,0,0,0,0,0,0,0,0,0,0]
	#just go through each line in the index-training file (each line represents a different piece of training data) and 
	# assign it a random language. 
	with open('TOEFL11-TRAIN/data/text/index-training.csv') as csv:
		for line in csv:
			response_info = re.split(',', line)
			num_responses_per_language[langs.index(response_info[2])] += 1
			#pick a random value r from 0 to len(langs)-1. langs[r] is this responses random assignment 
			random_classification = random.randint(0, len(langs)-1)
			#if the random guess was correct, update correct_assignemnts array
			if langs[random_classification] == response_info[2]:
				correct_assignments[random_classification] += 1
			else:
				false_positives[random_classification] += 1
	#evaluate accuracy, precision, recall
	print("Lang\tAcc\tPrec\tRec")
	for i in range(0, len(langs)):
		print(langs[i], end='\t')
		#accuracy
		print("%.3f" % (correct_assignments[i]/num_responses_per_language[i]), end='\t')
		#precision
		print("%.3f" % (correct_assignments[i]/(correct_assignments[i]+false_positives[i])), end='\t')
		#recall
		print("%.3f" % (correct_assignments[i]/(correct_assignments[i] + (num_responses_per_language[i]-correct_assignments[i]))))
	
naiveBaseline()
