# NLI
NLI Project for NLP

Anthony Sicilia
Brian Falkenstein
Yunkai Tang

All code written for Python 3.6.5  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-except evaluate.py, which was provided. This runs on python 2
q
Required Libraries  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Numpy 1.5.4  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Tensorflow 1.12.0 (1.11.0)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Keras 2.1.6-tf (2.2.4)
	
For testing, change the file paths for the TOEFL-11 data in 'constants.py', then run 'evaluation_setup.py' to get the 'predictions.csv' file containing the model's predictions. 
For cross validation / training the model, use 'continue_training.py'. The file paths must be set WITHIN 'continue_training.py'. 


For our reference, current trained_model.hdf5 was trained for 32 epochs, using exactly the params in the model.py file

