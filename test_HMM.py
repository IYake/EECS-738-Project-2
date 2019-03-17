import numpy as np
# https://en.wikipedia.org/wiki/Baum–Welch_algorithm
# 1) initialize A, B, and pi with random initial conditions, i.e., theta
# 2) forward procedure to find probability of seeing the sequence and being in state i and time t
#       - can also be used to predict next state, given sequence of observations
# 3) backward procedure to find probability of ending observation sequence given starting state i and time t
# 4) update transition and then emission matrices
#       - 4.1 calculate temporary variables lambda and epsilon
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
#                if i is 0 and t is num_obs - 2:
#                    print("i: %d" % i)
#                    print("j: %d" % j)
#                    print("B[j](t+1) = %f" % beta(j,t+1))
#                    print("A[i][j] = %f" % A[i][j])
#                    print("bj(yt+1) = %f" % B[j][B_cols.index(Y[t+1])])
#                    if (True):
#                        print("ans = %f" % ans, end = "\n\n")
            return ans
    
    betas = np.zeros([num_states,num_obs])
    for i in range(num_states):
        for t in range(num_obs):
            betas[i][t]=beta(i,t)
    return betas

betas = backward(Y,theta)
#print("betas: ", betas, end = "\n\n")

def update_lam(alphas,betas):
    lambdas = np.zeros([num_states,num_obs])
    #check that all i's and j's in program correspond to the correct number of states or observations
    denom = np.zeros([1,num_obs])
#    for i in range(num_symbols):
#        lambdas[j] = np.divide(
#                np.multiply(alphas[j],betas[j]),
#                np.sum(np.multiply(alphas[j],betas[j]))
#                )
    lambdas = np.multiply(alphas,betas)
    for j in range(num_states):
        denom += np.multiply(alphas[j],betas[j])
    lambdas = np.divide(lambdas,denom)
    #denominator is all one value because alpha*beta[0]+alpha*beta[1] is all the same value. Is this correct?
    return lambdas



lambdas = update_lam(alphas,betas)

#print("lambdas: ", lambdas[0], end = "\n\n")

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
            for t in range(num_obs-1):
                #range might be messed up
                epsilons[i][j][t] = epsilon(i,j,t)
    return epsilons
            
epsilons = update_eps(alphas,betas,Y,theta)
print(epsilons)

#for episilon i might be # of states, not number of observations.
# REDO all procedures with i and j as state variables, not number of observations!!!!!!!!




    
    

    