# Text-Sequence-LSTM

This repository can be applied to any type of script, medical, TV script, history, etc. It aims to generate text after training RNN type of networks using LSTM algorithm on text data so it become independent in predicting the rest of each sentence after using a prime word or phrase. 
The repository is divided into four main python scripts. 

### Helper script

This contains useful functions to load text data and preprocess them to be prepared for network training.

### Problem unit tests script

This has useful testing function to test data preprocessing codes and network building codes, to avoid complex bugs issues. 

### Network building script

This has all the codes needed to train an RNN network using Pytorch library. The hyperparameters can be adjusted based on the training performance.

### Testing script

This is Pytorch testing code block which generate text after giving the prime word. You can try different options based on the type of script you initially used for training. 

