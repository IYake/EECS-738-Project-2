#states = ('Healthy', 'Fever')
#end_st = 'E'
# 
#observations = ('normal', 'cold', 'dizzy')
# 
#start_prob = {'Healthy': 0.6, 'Fever': 0.4}
# 
#trans_prob = {
#   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
#   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
#   }
# 
#emm_prob = {
#   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
#   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
#   }

states = ('S1','S2')
end_st = 'end' #last observation reached
observations = ('N','N','N','N','N','E','E','N','N','N')
start_prob = {'S1':0.5,'S2':0.5}
trans_prob = {
        'S1' : {'S1':0.5,'S2':0.49,'end':.01 },
        'S2' : {'S1':0.3,'S2':0.69,'end':.01}
        }
emm_prob = {
        'S1' : {'N':0.3,'E':0.7},
        'S2' : {'N':0.8,'E':0.2}
        }

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

forward, backward, gammas = fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st)