from fwd_bck import trainHMM
import pandas as pd

def train():

    #read in the data
    df = pd.read_csv('frank_movie_lines.txt',header=None,sep=' \+\+\+\$\+\+\+ ', engine='python', usecols=[3,4])
    obs = df.values.tolist()
    obs = [tuple(x) for x in df.values]
    all_words = []

    #hyper parameters
    hidden_states = 15
    lines = len(obs)
    num_iters = 10
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
    trainHMM(hidden_states,all_words, num_iters, save_as = "frankenstein")

if __name__ == "__main__":
    train()
