import sys
# estimating the probability of the made observation sequence
# forward algorithm alpha-pass
# iteratively estimates the probability to be in a certain state i at time t
# having observed the observation sequence up to time t 

def forward_pass(A, B, pi, obs):
    
    T = len(obs)
    N = len(A)

    alpha = [[0] * N for _ in range(T)]
    
    # alpha_1[i] = b_i*(o_1)*π_i
    # loop over all possible states 
    for i in range(N):
        # initial state * observation probability given first observation
        alpha[0][i] = pi[i] * B[i][obs[0]]
        
    # at t we need to marginalize over the probability of having been in any other state at t-1 * matching observation probability
    for t in range(1, T):
        # loop over all possible states 
        for j in range(N):
            # b_i*(o_t)*sum[a_{j,i}*α_{t−1}(j)]
            alpha[t][j] = sum(alpha[t - 1][i] * A[i][j] for i in range(N)) * B[j][obs[t]]
            
    # compute the probability of having observed the final output sequence 
    return sum(alpha[T-1])

def convert_to_matrix(data, num_rows, num_cols):
    data = list(map(float, data))
    return [data[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]


# Read input
# Transition matrix
A_data = sys.stdin.readline().split()
# Observation matrix
B_data = sys.stdin.readline().split()
# initial state distribution 
pi_data = sys.stdin.readline().split()
obs_data = sys.stdin.readline().split()

# Convert into matrices
A = convert_to_matrix(A_data[2:], int(A_data[0]), int(A_data[1]))
B = convert_to_matrix(B_data[2:], int(B_data[0]), int(B_data[1]))
# initial states
pi = list(map(float, pi_data[2:]))
# the observed sequence
obs = list(map(int, obs_data[1:]))

# Calculate probability of observation sequence
prob = forward_pass(A, B, pi, obs)
print(round(prob, 6))
