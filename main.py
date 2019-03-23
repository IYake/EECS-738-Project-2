from fwd_bck import trainHMM, save, load
import pprint as pp
import string
import pandas as pd
import sys

def main():
    #read in the data
    col = "PlayerLine"
    df = pd.read_csv('Shakespeare_data.csv', usecols = [col])
    obs = df.values.tolist()
    obs = [tuple(x) for x in df.values]
    observations = ['N','N','N','N','N','E','E','N','N','N']
    observations = tuple(observations)
    all_words =[]
    for i in range(len(obs)):
       for j in range(len(obs[i])):
           sentence = obs[i][j].translate(str.maketrans('', '', string.punctuation)).split()
           if sentence[0].strip() == "SCENE":
               continue
           elif sentence[0].strip() == "ACT":
               continue
           else:
               for word in sentence:
                   all_words.append(word)
    all_words = tuple(all_words)
    model = trainHMM(2, all_words)
    pp.pprint(model)
    # model = load('train.pickle')
if __name__ == "__main__":
    main()
