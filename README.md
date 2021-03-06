# EECS-738-Project-2
Create a Hidden Markov Model to generate new text from a text corpus and perform text prediction given a sequence of words. 

# Training Approach
For this project we used the [Baum-Welch algorithm](https://en.m.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) to train a Hidden Markov Model. For numerical stability, we added scaling factors to the forward backward probabilities in the [forward-backward algorithm](https://en.m.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm).

The hyper parameters we have for the model are the number of hidden states, the max number of iterations if the algorithm hasn't converged to a certain likelihood, and the amount of training data read in.

# Generating text
1) A start state is randomly sampled from the start state probability matrix.
2) From the current state, the emission matrix is sampled to generate a word.
3) The next state is sampled according to the probability distribution of the current state in the transition matrix.
4) Repeat steps 2-3 until the prompted number of words is generated.
5) Display the sequence of words.

## Sample Output
```
How many words to generate?
15

Wonderful. if you eye Very
I.: evil? Jenner. as It 
NEVER: Waldman. the 
CREATURE: talent
```

# Predicting text
1) Use the [Viterbi algorithm](https://en.m.wikipedia.org/wiki/Viterbi_algorithm) on the user-entered text sequence to find the most likely last state for the HMM to be in.
2) Do steps 2-5 of the generating text section.

Note: If the user enters a word that isn't found in the training dataset, then we don't predict any following sequence of text.

## Sample Output
```
Enter your sequence of words: 

CREATURE

How many words to predict? 
50

removed a in mind. can He letter have Oh, He may your the It's and Like 
HENRY: She 
CREATURE: and 
VICTOR: To the questions? 
SHE: heard God held my you He I you my expense north, hurt He on? come missionaries. done. You're Your always took Frankenstein? 
JUSTINE: 
CREATURE: mad?
```
# Dataset:
For our project we chose to use the quotes from the movie Frankenstein (1931) in the [Cornell Movie Dataset](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). 

# How to Compile:
Our project uses Python 3.7.1 with the modules: pickle, random, sys, math, tqdm, pandas.
To run this project, run `python main.py`. This command will prompt the user choose
to either 1) generate n new words of text or 2) predict the next n words
of text following the user's input. 

# Limitations
- The model was not trained using numpy matrices so it's very slow. 
- Training on hidden states at around 75 or greater causes underflow errors because the probabilities in the transition matrix become too small.
- There is semblance of probabilistic understanding in the order of words that are generated but usually nothing coherent comes out.

# References 
https://en.m.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

https://en.m.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

https://en.m.wikipedia.org/wiki/Viterbi_algorithm

https://pdfs.semanticscholar.org/4ce1/9ab0e07da9aa10be1c336400c8e4d8fc36c5.pdf
