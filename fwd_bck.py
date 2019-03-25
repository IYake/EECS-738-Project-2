"""
Forward backward algorithm from:
https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm#Python_example
"""
import pprint as pp
import pickle
import random
import sys
import math
#import time
from tqdm import tqdm

def trainHMM(num_states, observations, save_as = "train"):
    num_states = num_states
    observations = observations
    string = 'S'
    states = [string+str(i) for i in range(1, num_states+1)]
    end_st = 'end'

    ob_type = list(dict.fromkeys(observations).keys())


    #pi
    start_prob = {}
    #randomizing the inintal values of start_prob aka pi
    start_prob_sum = 0
    for i in range(len(states)):
        start_prob[states[i]]= random.randrange(0,1000)
        start_prob_sum += start_prob[states[i]]
    for i in range(len(states)):
        start_prob[states[i]] = start_prob[states[i]]/start_prob_sum

    trans_prob = {}
    trans_prob_sum = [0.01 for _ in range(len(states))]
    for i in range(len(states)):
        trans_prob[states[i]] = {}
        for j in range(len(states)+1):
            if j == len(states):
                trans_prob[states[i]][end_st] = .01
            else:
                trans_prob[states[i]][states[j]] = random.randrange(0,1000)
                trans_prob_sum[i] += trans_prob[states[i]][states[j]]
    for i in range(len(states)):
        for j in range(len(states)):
            trans_prob[states[i]][states[j]] = trans_prob[states[i]][states[j]]/trans_prob_sum[i]

    # example:
    # trans_prob = {
    #         'S1' : {'S1':0.5, 'S2':0.49,'end':.01 },
    #         'S2' : {'S1':0.8, 'S2':0.19,'end':.01 }
    #         }
    emm_prob = {}
    for i in range(len(states)):
        emm_prob[states[i]] = {}
        for j in range(len(ob_type)):
            emm_prob[states[i]][ob_type[j]] = (1/len(ob_type))
    #
    # emm_prob = {
    #         'S1' : {'N':0.5,'E':0.5},
    #         'S2' : {'N':0.5,'E':0.5}
    #         }
    #is this fwd_bck necessary?
    forward, backward, gammas, likelihood = fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st)


    model = Update(observations, trans_prob, emm_prob, start_prob, states,end_st,ob_type)
    save(model, file_prefix = save_as)
    return model


def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    # forward part of the algorithm
    c = []
    fwd = []
    likelihood = 0
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)
            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum
        #scaling value
        c.append(0)
        if i == len(observations)-1:
            for st in states:
                likelihood += f_curr[st]
        for st in states:
            c[i] += f_curr[st]
        for st in states:
            f_curr[st] = f_curr[st] / c[i]

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # backward part of the algorithm
    bkw = []
    b_prev = {}
    c_test_for_b = []
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        c_test_for_b.append(0)
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)
        #scaling value
        for st in states:
            c_test_for_b[i] += b_curr[st]
        for st in states:
            b_curr[st] = b_curr[st] / c_test_for_b[i]
        bkw.insert(0,b_curr)
        b_prev = b_curr

    # p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)
    # merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    #assert p_fwd == p_bkw
    return fwd, bkw, posterior,likelihood

#https://en.wikipedia.org/wiki/Viterbi_algorithm
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t-1][states[0]]["prob"]*trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"]*trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
                    
#    for line in dptable(V):
#        print(line)
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

#    print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)
    return opt

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
# Maximation functions

def update_start_prob(gammas):
    return gammas[0]

#get epsilon
def update_eps(forward,backward,observations,trans_prob,emm_prob,states):
    epsilons = {st1 : {st2 : {i : 0 for i in range(len(observations))} for st2 in states} for st1 in states}

    denom = {i : 0 for i in range(len(observations))}
    for st1 in states:
        for st2 in states:
            for i in range(len(observations)-1):
                denom[i] += forward[i][st1]*trans_prob[st1][st2]              \
                      *backward[i+1][st2]*emm_prob[st2][observations[i+1]]
    
    for st1 in states:
        for st2 in states:
            for i in range(len(observations)-1):#can't assign last value
                eps = forward[i][st1]*trans_prob[st1][st2]              \
                      *backward[i+1][st2]*emm_prob[st2][observations[i+1]]
                epsilons[st1][st2][i] = eps / denom[i]
    return epsilons

def update_trans_prob(epsilons,gammas,states,observations):
    temp_trans = {st1:{st2:0 for st2 in states}for st1 in states}
    denom = {st:0 for st in states}
    for st in states:
        for i in range(len(observations)):
            denom[st] += gammas[i][st]

    for st in states:
        if 0 in denom.keys():
            print("denom has zero")
            sys.quit()

    for st1 in states:
        for st2 in states:
            temp_trans[st1][st2] = 0
            for i in range(len(observations)-1):
                pass
                temp_trans[st1][st2] += epsilons[st1][st2][i]
            temp_trans[st1][st2] /= denom[st1]
        
    for st in states:
        temp_trans[st]['end'] = 0.01

#normalize
    for st1 in states:
        rowSum = 0
        for st2 in states:
            rowSum += temp_trans[st1][st2]
        for st2 in states:
            temp_trans[st1][st2] /= rowSum
    return temp_trans

def update_em_prob(observations, gammas,states,ob_type):
    temp_em = {st1:{ob:0 for ob in ob_type} for st1 in states}
    denom = {st:0 for st in states}
    #denominator
    for st in states:
        for i in range(len(observations)):
            denom[st] += gammas[i][st]
    #numerator
    for st1 in states:
        for ob in ob_type:
            for i in range(len(observations)):
                if observations[i] == ob:
                    temp_em[st1][ob] += gammas[i][st1]
            temp_em[st1][ob] /= denom[st1]
    return temp_em

def Update(obs, trans, emm, start, sts, end,ob_types):
    #do this to log probability in the future
    print("Training:")
    forward = None
    backward = None
    gammas = None
    likelihood = None
    epsilons = None
    observations = obs
    trans_prob = trans
    emm_prob = emm
    start_prob = start
    states = sts
    end_st = end
    ob_type = ob_types
    for i in tqdm(range(100)):
        forward, backward, gammas, likelihood = fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st)
        epsilons = update_eps(forward, backward, observations, trans_prob, emm_prob, states)
        trans_prob = update_trans_prob(epsilons, gammas,states,observations)
        emm_prob = update_em_prob(observations, gammas,states,ob_type)
        start_prob = update_start_prob(gammas)
    #print("Likelihood: ", likelihood) #need to scale this back
    #return trained model
    return [trans_prob,emm_prob,start_prob,states,ob_type, likelihood]

def save(model, file_prefix = 'train'):
    with open(file_prefix+'.pickle','wb') as f:
        pickle.dump(model,f)

def load(filename):
    with open(filename,'rb') as f:
        model = pickle.load(f)
    return model

def generate(model, numWords):
    trans_prob = model[0]
    emm_prob = model[1]
    start_prob = model[2]
    states = model[3]
    ob_type = model[4]

    emm_list = [[0 for j in range(len(ob_type))] for i in range(len(states))] #list form to be interpretated by choices()
    for st in states:
        i = states.index(st)
        for ob in ob_type:
            j = ob_type.index(ob)
            emm_list[i][j] = emm_prob[st][ob]

    trans_list = [[0 for j in range(len(states))] for i in range(len(states))] #list form to be interpretated by choices()
    for st1 in states:
        i = states.index(st1)
        for st2 in states:
            j = states.index(st2)
            trans_list[i][j] = trans_prob[st1][st2]

    #find starting states
    start_prob_list = list(start_prob.values())
    start_st = random.choices([i for i in range(len(start_prob_list))], start_prob_list,k=1)[0]

    generatedSentence = []
    curr_st = start_st
    for i in range(numWords):
        generatedSentence.append(random.choices(ob_type,emm_list[curr_st],k=1)[0])
        curr_st = random.choices([i for i in range(len(trans_list))], trans_list[curr_st],k=1)[0]


    print(" ".join(generatedSentence))
    
#given a sequence of text, predict the words that come after
#using the trained model
def predict(model,numWords,observations):
    trans_prob = model[0]
    emm_prob = model[1]
    start_prob = model[2]
    states = model[3]
    ob_type = model[4]
    
    #make lists from dictionaries to predict text if necessary
    emm_list = [[0 for j in range(len(ob_type))] for i in range(len(states))] #list form to be interpretated by choices()
    for st in states:
        i = states.index(st)
        for ob in ob_type:
            j = ob_type.index(ob)
            emm_list[i][j] = emm_prob[st][ob]

    trans_list = [[0 for j in range(len(states))] for i in range(len(states))] #list form to be interpretated by choices()
    for st1 in states:
        i = states.index(st1)
        for st2 in states:
            j = states.index(st2)
            trans_list[i][j] = trans_prob[st1][st2]

#    #find starting states
#    start_prob_list = list(start_prob.values())
#    start_st = random.choices([i for i in range(len(start_prob_list))], start_prob_list,k=1)[0]
    
    
    wordsInData = True
    for i in range(len(observations)):
        if observations[i] not in list(emm_prob[states[0]].keys()):
            print("\""+observations[i]+"\" not seen in data set. Try again\n")
            wordsInData = False
            break
        
    if wordsInData:
     #find most likely sequence of states 
        opt = viterbi(observations,states,start_prob,trans_prob,emm_prob)
        last_state = opt[-1]
        
        prediction = []
        curr_st = int(last_state[1:]) #remove 'S' and convert to int
        for i in range(numWords):
            prediction.append(random.choices(ob_type,emm_list[curr_st],k=1)[0])
            curr_st = random.choices([i for i in range(len(trans_list))], trans_list[curr_st],k=1)[0]
        
        print(" ".join(prediction))

    
if __name__ == "__main__":
    observations = tuple([random.choices(['N','E'],[8,2],k=1)[0] for i in range(10)])
    model = trainHMM(2,observations, save_as = "testModel")

    generate(model,10)

