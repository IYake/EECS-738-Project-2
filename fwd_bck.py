"""
Forward backward algorithm from:
https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm#Python_example
"""
import numpy as np
import pprint

def Exe(num_states, observations):
    # num_states = 2 #taken in with HMM parameters
    num_states = num_states
    observations = observations
    # observations = ('N','N','N','N','N','E','E','N','N','N')

    string = 'S'
    states = [string+str(i) for i in range(1, num_states+1)]
    # states = ('S1','S2')
    end_st = 'end'

    ob_type = ["N", "E"]
    # ob_type = list(dict.fromkeys(observations).keys())
    #pi
    start_prob = {'S1':0.5,'S2':0.5}
    start_prob = {}
    for i in range(len(states)):
        start_prob[states[i]]= (1/num_states)

    trans_prob = {}
    for i in range(len(states)):
        trans_prob[states[i]] = {}
        for j in range(len(states)+1):
            if j == len(states):
                trans_prob[states[i]][end_st] = .01
            elif j== len(states) -1:
                trans_prob[states[i]][states[j]] = ((1/num_states)-.01)
            else:
                trans_prob[states[i]][states[j]] = (1/num_states)
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
    forward, backward, gammas = fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st)
    print("Before update")
    Update(observations, trans_prob, emm_prob, start_prob)


def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    # forward part of the algorithm
    fwd = []
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

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # backward part of the algorithm
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    #p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    #assert p_fwd == p_bkw
    return fwd, bkw, posterior
    #return posterior


############### MAXIMIZATION STEP ##################

def update_start_prob(gammas):
    return gammas[0]

#get epsilon
def update_eps(forward,backward,observations,trans_prob,emm_prob):
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

def update_trans_prob(epsilons,gammas):
    temp_trans = {st1:{st2:0 for st2 in states}for st1 in states}
    denom = {st:0 for st in states}
    for st in states:
        for i in range(len(observations)):
            denom[st] += gammas[i][st]

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

def update_em_prob(observations, gammas):
    temp_em = {st1:{ob:0 for ob in ob_type}for st1 in states}
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


def Update(observations, trans_prob, emm_prob, start_prob):
    #do this to log probability in the future
    for i in range(10):
        # print(i)
        forward, backward, gammas = fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st)
        epsilons = update_eps(forward, backward, observations, trans_prob, emm_prob)
        trans_prob = update_trans_prob(epsilons, gammas)
        emm_prob = update_em_prob(observations, gammas)
        start_prob = update_start_prob(gammas)
#        print("Iteration: %i" % (i+1))
#        print("pi: ", start_prob)
#        pprint.pprint(trans_prob)
#        print("\n")
        print("EMM")
        pprint.pprint(emm_prob)
        print("Trans")
        pprint.pprint(trans_prob)
