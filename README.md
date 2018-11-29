# NLI
NLI Project for NLP

Anthony Sicilia
Brian Falkenstein
Yunkai Tang

All code written for Python 3.6.5\\
	-except evaluate.py, which was provided. This runs on python 2\\

Required Libraries\\
	-Numpy 1.5.4\\
	-Tensorflow 1.12.0 (1.11.0) \\
	-Keras 2.1.6-tf (2.2.4)  \\
	
For testing, change the file paths for the TOEFL-11 data in 'constants.py', then run 'evaluation_setup.py' to get the 'predictions.csv' file containing the model's predictions. 
For cross validation / training the model, use 'continue_training.py'. The file paths must be set WITHIN 'continue_training.py'. 

Useful links:
NLTK api: http://www.nltk.org/api/nltk.html
Character encoding: https://offbit.github.io/how-to-read/
