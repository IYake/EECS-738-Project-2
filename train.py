from fwd_bck import trainHMM, save
import pprint as pp
import string
import pandas as pd

def train():

    #read in the data
    col = "PlayerLine"
    df = pd.read_csv('Shakespeare_data.csv', usecols = [col])
    obs = df.values.tolist()
    obs = [tuple(x) for x in df.values]
    all_words = []

    #hyper parameters
    hidden_states = 5
    lines = 100
    for i in range(lines):#range(len(obs)):
       for j in range(len(obs[i])):
           sentence = obs[i][j].split()
           if sentence == []:
               continue
           if sentence[0].strip() == "SCENE":
               continue
           elif sentence[0].strip() == "ACT":
               continue
           else:
               for word in sentence:
                   all_words.append(word)
    all_words = tuple(all_words)
    #trainHMM(hidden_states,all_words, save_as = "model"+str(hidden_states)+"_"+str(lines))
    trainHMM(hidden_states,all_words, save_as = "hmm")
   # model = load("model"+str(hidden_states)+str(lines)+".pickle")
    #generate(model,15)

if __name__ == "__main__":
    train()
