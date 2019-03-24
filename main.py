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
    all_words =[]
    for i in range(20):#range(len(obs)):
       for j in range(len(obs[i])):
           sentence = obs[i][j].translate(str.maketrans('', '', string.punctuation)).split()
           # print(sentence)
           if sentence == []:
               continue
           if sentence[0].strip() == "SCENE":
               continue
           elif sentence[0].strip() == "ACT":
               continue
           else:
               for word in sentence:
                   all_words.append(word.lower())
#    print(all_words)
#    for word in all_words:
#        if word == "":
#            print("")
    all_words = tuple(all_words)
    model = trainHMM(2,all_words, save_as = "test_jg")
    model = load('test_jg.pickle')
    generate(model,100)
    # model = load('train.pickle')
if __name__ == "__main__":
    main()
