import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #this removes unwanted system warning messages
import tensorflow as tf
import sys

# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500 # we add zero padding for shorter reviews so review length now is constant 500 words
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words)) #32 dimensional vector with 500 inputs
model.add(Flatten()) #need to flatten 500X32 sized matix
model.add(Dense(250, activation='relu'))  # use of dense hidden layer of 250 units with RELU as activation function
model.add(Dense(1, activation='sigmoid')) # sigmoid to output as 0 or 1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)

# since this model overfits very quickly so we use only 2 epochs. Overfitting leads to better train results and bad test results
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
