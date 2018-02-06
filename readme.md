Learning how to predict the sentiment of a review through IMDB sentiment Analysis as "Positive" or "Negative"

Topics Covered
    About the IMDB sentiment analysis problem for natural language processing and how to load it in Keras.
    How to use word embedding in Keras for natural language problems.
    How to develop and evaluate a multi-layer perception model for the IMDB problem.
    How to develop a one-dimensional convolutional neural network model for the IMDB problem.

This dataset has 25000 highly polar (good or bad) movie reviews for training.
Similarly we test for another 25000 dataset for the movie review analyzer.

We implement both model
1. Simple model through which we get accuracy of almost 87%.
2. 1-D CNN model improves the accuracy upto 88% will use other libraries like textblob and see its effect on the accuracy.

Credits https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

IMDB Dataset used from http://ai.stanford.edu/~amaas/data/sentiment/


Also checked the textblob Library for the different NLP tasks that it can provide future work includes implementation of the sentiment analysis using the textblob Library.

The top 5 libraries for NLP tasks as per https://elitedatascience.com/python-nlp-libraries
The Conqueror: NLTK                      http://www.nltk.org/
The Prince: TextBlob                     https://textblob.readthedocs.io/en/dev/index.html
The Mercenary: Stanford CoreNLP          https://stanfordnlp.github.io/CoreNLP/
The Usurper: spaCy                       https://github.com/explosion/spaCy
The Admiral: gensim                      https://pypi.python.org/pypi/gensim
