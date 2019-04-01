from hmm import trainHMM
import pandas as pd
import sys

def train():
    if len(sys.argv) != 3:
        raise Exception('Must enter 2 parameters: 1)data file and 2) output model name')
    #store arguments
    data_src = sys.argv[1]
    data_out = sys.argv[2]
    #read in the data
    df = pd.read_csv(data_src,header=None,sep=' \+\+\+\$\+\+\+ ', engine='python', usecols=[3,4])
    obs = df.values.tolist()
    obs = [tuple(x) for x in df.values]
    all_words = []

    #hyper parameters
    hidden_states = 75
    lines = len(obs)
    num_iters = 1000
    speakers = []
    for i in range(lines):
       for j in range(len(obs[i])):
           sentence = obs[i][j].split()
           if sentence == []:
               continue
           else:
               for word in sentence:
                   all_words.append(word)
       if obs[i][0] not in speakers:
           speakers.append(obs[i][0])
    all_words = tuple(all_words)
    if data_out == "DEFAULT_MODEL":
        raise Exception('Cannot write over the default file')
    trainHMM(hidden_states,all_words, num_iters, save_as = data_out)

if __name__ == "__main__":
    train()
