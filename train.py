from fwd_bck import trainHMM, save
import pprint as pp
import string
import pandas as pd
import re

def train():

    #read in the data
    # col = "PlayerLine"
    # df = pd.read_csv('Shakespeare_data.csv', usecols=[col])
    df = pd.read_table('movie_lines.txt',header=None,sep=' \+\+\+\$\+\+\+ ', engine='python', usecols=[3,4])
    # pp.pprint(df)
    obs = df.values.tolist()
    obs = [tuple(x) for x in df.values]
    all_words = []

    #hyper parameters
    hidden_states = 20
    lines = 100
    num_iters = 200
    speakers = []
    for i in range(lines):#range(len(obs)):
       for j in range(len(obs[i])):
           sentence = obs[i][j].split()
           if sentence == []:
               continue
           # if sentence[0].strip() == "SCENE":
           #     continue
           # elif sentence[0].strip() == "ACT":
           #     continue
           else:
               for word in sentence:
                   all_words.append(word)
       if obs[i][0] not in speakers:
           speakers.append(obs[i][0])
    all_words = tuple(all_words)
    #trainHMM(hidden_states,all_words, save_as = "model"+str(hidden_states)+"_"+str(lines))
    trainHMM(hidden_states,all_words, num_iters, save_as = "hmm")
   # model = load("model"+str(hidden_states)+str(lines)+".pickle")
    #generate(model,15)

# if __name__ == "__main__":
#     # train()
train()
