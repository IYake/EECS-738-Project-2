# EECS-738-Project-2
Create a Hidden Markov Model to generate new text from a text corpus and perform text prediction given a sequence of words. 

# Approach
For this project we used the Baum-Welch algorithm to train a Hidden Markov Model. Once the model was trained, we used the transition matrix and the emission matrix to predict the next n number of words. 

# Dataset:
http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

# How to Compile:
Our project uses Python 3.7.1 with the modules: pickle, random, sys, math, tqdm, pandas.
To run this project, run `python main.py`. This command will propt the user choose
the functionality they desire: 1) generate n new words of text or 2) predict the next n words
of text based on the input word(s). Based on the user's selection, the program will
output the prediction for the next n words. The output also takes into consideration
who in the movie was speaking. If the user chooses the second option and enters a word or 
words that are not in the Hidden Markov Model, then the model cannot calculate the next word
and therefore the program reprompts the user. 

# References 
https://en.m.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
https://en.m.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
https://en.m.wikipedia.org/wiki/Viterbi_algorithm
