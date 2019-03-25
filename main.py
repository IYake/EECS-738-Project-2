from fwd_bck import trainHMM, save, load, generate
import pprint as pp
import string
import pandas as pd
#import sys
import math

def main():
    
    #read in the data
    col = "PlayerLine"
    df = pd.read_csv('Shakespeare_data.csv', usecols = [col])
    obs = df.values.tolist()
    obs = [tuple(x) for x in df.values]
    observations = ['N','N','N','N','N','E','E','N','N','N']
    observations = tuple(observations)
    all_words = []
    
    #hyper parameters
    hidden_states = 5
    lines = 100
    for i in range(lines):#range(len(obs)):
       for j in range(len(obs[i])):
           sentence = obs[i][j].translate(str.maketrans('', '', string.punctuation)).split()
           if sentence == []:
               continue
           if sentence[0].strip() == "SCENE":
               continue
           elif sentence[0].strip() == "ACT":
               continue
           else:
               for word in sentence:
                   all_words.append(word.lower())
    all_words = tuple(all_words)
    model = trainHMM(hidden_states,all_words, save_as = "model"+str(hidden_states)+str(lines))
    model = load("model"+str(hidden_states)+str(lines)+".pickle")
    generate(model,15)
if __name__ == "__main__":
    main()
