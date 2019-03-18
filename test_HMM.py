import numpy as np
# https://en.wikipedia.org/wiki/Baum–Welch_algorithm
# 1) initialize A, B, and pi with random initial conditions, i.e., theta
# 2) forward procedure to find probability of seeing the sequence and being in state i and time t
#       - can also be used to predict next state, given sequence of observations
# 3) backward procedure to find probability of ending observation sequence given starting state i and time t
# 4) update transition and then emission matrices
#       - 4.1 calculate temporary variables gamma and epsilon
#       - 4.2 update pi, then transition matrix, then emission matrix
# 5) repeat 1-4 until desired level of convergence (using log likelihood function)
#       - also can test with diff number of states to see best convergence


# Example, checking if a chicken legs eggs that noon, every day

# ---------------------------------------------------------------


# observation
Y = ["N","N","N","N","N","E","E","N","N","N"]
B_cols = ["N","E"]
num_symbols = len(dict.fromkeys(Y))
num_obs = len(Y)
#arbitrary symbol length. Usually 1 character
symbol_len = 1
Y_trans = []
s = None
for i in range(len(Y)-2*symbol_len):
    s = ""
    for j in range(2*symbol_len):
        s += Y[i+j]
    Y_trans.append(s)

# initialization (guess these)
# transition matrix
A = np.array([[0.5,0.5],[0.3,0.7]])
# emmission matrix
#   - rows are states and columns are observation types
B = np.array([[0.3,0.7],[0.8,0.2]])
#initial state distribution
pi = np.array([[0.2],[0.8]])
#arbitrary number of states
num_states = 2

theta = [A,B,pi]

def forward(Y,theta):
    #return alpha, the probability of seeing sequence Y, and being in state i at time t
    # use dynamic programming after this works
    A = theta[0]
    B = theta[1]
    pi = theta[2]
#    print(pi[0])
    def alpha(i,t):
        if t is 0:
            return pi[i]*B[i][B_cols.index(Y[1])]
        else:
            ans = 0
            for j in range(num_states):
                ans += alpha(j,t-1)*A[j][i]*B[i][B_cols.index(Y[t])]
            return ans
    #use index 0 as t=1
    alphas = np.zeros([num_states,num_obs])
    for i in range(num_states):
        for t in range(num_obs):
            alphas[i][t]=alpha(i,t)
    return alphas

alphas = forward(Y,theta)
#print("alphas: ", alphas, end="\n\n")
def backward(Y,theta):
    A = theta[0]
    B = theta[1]

    def beta(i,t):
        if t is (num_obs-1):
            #this might not be the correct interpretation
            return 1
        else:
            ans = 0
            for j in range(num_states):
                ans += beta(j,t+1)*A[i][j]*B[j][B_cols.index(Y[t+1])]
            return ans

    betas = np.zeros([num_states,num_obs])
    for i in range(num_states):
        for t in range(num_obs):
            betas[i][t]=beta(i,t)
    return betas

betas = backward(Y,theta)
#print("betas: ", betas, end = "\n\n")

def update_gam(alphas,betas):
    gammas = np.zeros([num_states,num_obs])
    #check that all i's and j's in program correspond to the correct number of states or observations
    denom = np.zeros([1,num_obs])
#    for i in range(num_symbols):
#        gammas[j] = np.divide(
#                np.multiply(alphas[j],betas[j]),
#                np.sum(np.multiply(alphas[j],betas[j]))
#                )
    gammas = np.multiply(alphas,betas)
    #print("gammas added: ", gammas[0]+gammas[1], end="\n\n")
    for j in range(num_states):
        denom += np.multiply(alphas[j],betas[j])
    #print("denom: ",denom, end="\n\n")
    gammas = np.divide(gammas,denom)

    #denominator is all one value because alpha*beta[0]+alpha*beta[1] is all the same value. Is this correct?
    return gammas



gammas = update_gam(alphas,betas)
print(np.sum(gammas[0]))
print(np.sum(gammas[1]))
#print("\ngammas: ", gammas, end = "\n\n")

#Check if this works. Not sure
def update_eps(alphas,betas,Y,theta):
    A = theta[0]
    B = theta[1]

    epsilons = np.zeros([num_states,num_states,num_obs])
    def epsilon(i,j,t):
        eps = alphas[i][t]*A[i][j]*betas[j][t+1]*B[j][B_cols.index(Y[t+1])]
        denom = 0
        for i in range(num_states):
            for j in range(num_states):
                denom += alphas[i][t]*A[i][j]*betas[j][t+1]*B[j][B_cols.index(Y[t+1])]
        return eps/denom


    for i in range(num_states):
        for j in range(num_states):
            for t in range(num_obs):
                #if statement placed because equation can't parse last value, bj(yt+1) doesn't exist
                if t != num_obs-1:
                    epsilons[i][j][t] = epsilon(i,j,t)
                else:
                    epsilons[i][j][t] = None
    return epsilons

epsilons = update_eps(alphas,betas,Y,theta)
#print(epsilons)

def update_pi(gammas):
    temp = np.zeros([num_states,1])
    for i in range(num_states):
        temp[i] = gammas[i][0]
    return temp

#update emission matrix

def update_B(Y,gammas,B_cols):
    temp = np.zeros([num_states,num_symbols])
    denom = np.zeros([num_states,1])
    for i in range(num_states):
        denom[i] = np.sum(gammas[i])

    for symbol in enumerate(B_cols):
        for i in range(num_states):
            for t in range(num_obs):
                if Y[t] == symbol[1]:
                    temp[i][symbol[0]] += gammas[i][t]
            temp[i][symbol[0]] /= denom[i]
            #print(temp)
    for r in range(num_states):
        for c in range(num_symbols):
            temp[r][c] = temp[r][c]/np.sum(temp[r])
    print(temp)
    return temp

B = update_B(Y,gammas,B_cols)
#print(B)
#print(gammas)

def update_A(epsilons,gammas):
    denom = np.zeros([num_states,1])
    for i in range(num_states):
        denom[i] = np.sum(gammas[i])

    temp = np.zeros([num_states,num_states])
    for i in range(num_states):
        for j in range(num_states):
            temp[i][j] = np.sum(epsilons[i][j][:-1])/denom[i]

    #normalize rows
    for r in range(num_states):
        for c in range(num_states):
            temp[r][c] = temp[r][c]/np.sum(temp[r])
    return temp

A = update_A(epsilons,gammas)
def Update(Y, theta):
    for i in range(10):
        # print(i)
        alphas = forward(Y,theta)
        betas = forward(Y,theta)
        gammas = update_gam(alphas, betas)
        epsilons = update_eps(alphas, betas, Y, theta)
        A = update_A(epsilons, gammas)
        B = update_B(Y, gammas, B_cols)
        pi = update_pi(gammas)
        theta[0] = A
        theta[1] = B
        theta[2] = pi
        print("LOOP")
#        print("gammas: " , gammas)
#        print("alphas: ", alphas)
#        print("betas:", betas)
#        print("pi: ", pi)
        print("pi: ",)
        print("\n")
#Update(Y, theta)
