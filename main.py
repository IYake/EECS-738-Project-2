from fwd_bck import Exe
import pprint as pp
import pandas as pd
import sys

def main():
    #read in the data
    col = "PlayerLine"
    df = pd.read_csv('Shakespeare_data.csv', usecols = [col])
    obs = df.values.tolist()
    # obs = [tuple(x) for x in df.values]
    # observations = ('N','N','N','N','N','E','E','N','N','N')
    all_words =[]
    for i in range(10):#range(len(obs)):
        for j in range(len(obs[i])):
            print(obs[i][j])
            sentence = obs[i][j].split()
            for word in sentence:
                all_words.append(word)

    # pp.pprint(all_words)
    Exe(2, all_words)
if __name__ == "__main__":
    main()
