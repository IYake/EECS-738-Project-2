from fwd_bck import Exe
import pprint as pp
import pandas as pd
import sys

def main():
    #read in the data
    col = "PlayerLine"
    df = pd.read_csv('Shakespeare_data.csv', usecols = [col])
    obs = df.values.tolist()
    all_words =[]
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            print(obs[i][j])
            sentence = obs[i][j].split()
            for word in sentence:
                all_words.append(word)

    Exe(2, all_words)
if __name__ == "__main__":
    main()
