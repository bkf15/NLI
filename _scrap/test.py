import nltk
from nltk import bigrams
from nltk.collocations import *

#f = open('test_text.txt')
#str = f.read().replace('\n', ', ')
str = 'I would like an ice cream please. I am hungry. I would like to not be hungry.'
str2 = 'ice cream is a dumb dessert and no one should eat it, even if they are hungry.'

##collections. check http://www.nltk.org/howto/collocations.html or even better http://www.nltk.org/api/nltk.html?highlight=freqdist
#bigram_measures = nltk.collocations.BigramAssocMeasures()
#finder = BigramCollocationFinder.from_words(str.split())
##removes bigrams with frequency < 3
#finder.apply_freq_filter(3)
##gets the top 4 bigrams from the text, prints them
#print(finder.nbest(bigram_measures.pmi, 4))

##does some probability stuff on the ngrams
#scored = finder.score_ngrams(bigram_measures.raw_freq)
#print(scored)
#s = 0
#for i in range(len(scored)):
#	s += scored[i][1]
#print(s)
#note that the scores don't sum to 1

fd = nltk.FreqDist(str.split())
print(list(fd))

#bigrams = list(bigrams("Hello my name is brian. Hello my name is chad.".split()))
#print(bigrams)