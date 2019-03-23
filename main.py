from fwd_bck import Exe
import pprint as pp
import pandas as pd
import string
import sys

def main():
    #read in the data
    col = "PlayerLine"
    df = pd.read_csv('Shakespeare_data.csv', usecols = [col])
    obs = df.values.tolist()
    # obs = [tuple(x) for x in df.values]
    # observations = ('N','N','N','N','N','E','E','N','N','N')

    # print(df)
    # print((tuple(obs[:][1])))
    # pp.pprint(tuple(obs))
    print(type(tuple(obs[0][0])))
    obs = tuple(obs)
    # for e in obs:
    #     obs[e] = ''.join(obs[e])
    # pp.pprint(obs)
    table = str.maketrans("", "", string.punctuation)
    print(table)

    # Exe(2, obs)
if __name__ == "__main__":
    main()
