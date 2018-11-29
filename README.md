# NLI
NLI Project for NLP

Anthony Sicilia  
Brian Falkenstein  
Yunkai Tang  

All code written for Python 3.6.5  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-except evaluate.py, which was provided. This runs on python 2

Required Libraries  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Numpy 1.5.4  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Tensorflow 1.12.0 (1.11.0)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Keras 2.1.6-tf (2.2.4)

IMPORTANT NOTE: Some layers hash values, for non-random results all programs should be run as follows:  
PYTHONHASHSEED=0 python3 <program_name>.py

The procedure for testing our pretrained model:
1) Run 'evaluation_setup.py' REMEMBER python hashseed.
2) Run 'evaluate.py' on csv_out_file (in constants.py)

The procedure for creating, training, and evalutating a new model (i.e. for cross-validation):  

1) Set file paths in 'constants.py' for training and development sets
2) Run 'continue_training.py' (REMEMBER python hashseed). Note, you may want to set the model_to_load and save_path_for_model in 'constants.py'. There is a default 'untrained_model.hdf5' provided that is built exactly as our pre-trained model, but isn't trained. See model.py if you would like to create your own.

3) Reset model_to_load in 'constants.py' to be model trained in 'continue_training.py' and run 'evaluation_setup.py' (REMEMBER python hashseed).

Note: You may want to modify the number of epochs in 'continue_training.py'. It is currently set to 35. Our best model took 32 epochs (between 5 and 6 hours). See NOTE in 'continue_training.py' for details.

