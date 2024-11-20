import sys

# most likely sequence of hidden states given the observations

def viterbi_algorithm(A, B, pi, obs):
    T = len(obs)
    N = len(A)

    # Initialize delta matrices
    delta = [[0 for _ in range(N)] for _ in range(T)]
    delta_id = [[0 for _ in range(N)] for _ in range(T)]

    # Initialize first time step
    # compute  probability of having observed o1:t and being in a state Xt = xi given the most likely preceding state Xt−1 = xj for each t
    for i in range(N):
        delta[0][i] = pi[i] * B[i][obs[0]]
        delta_id[0][i] = 0

    # update delta 
    # Fill delta matrices for all time steps
    for t in range(1, T):
        for i in range(N):
            max_prob = -1
            max_id = -1
            for j in range(N):
                prob = delta[t - 1][j] * A[j][i]
                if prob > max_prob:
                    max_prob = prob
                    max_id = j
            delta[t][i] = max_prob * B[i][obs[t]]
            delta_id[t][i] = max_id

    # Backtrack to find most likely state sequence
    seq = [-1] * T
    seq[T - 1] = max(range(N), key=lambda i: delta[T - 1][i])
    for t in range(T - 2, -1, -1):
        seq[t] = delta_id[t + 1][seq[t + 1]]

    return seq


def convert_to_matrix(data, num_rows, num_cols):
    data = list(map(float, data))
    return [data[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]


# Read input
A_data = sys.stdin.readline().split()
B_data = sys.stdin.readline().split()
pi_data = sys.stdin.readline().split()
obs_data = sys.stdin.readline().split()

# Convert into matrices
A = convert_to_matrix(A_data[2:], int(A_data[0]), int(A_data[1]))
B = convert_to_matrix(B_data[2:], int(B_data[0]), int(B_data[1]))
pi = list(map(float, pi_data[2:]))
obs = list(map(int, obs_data[1:]))

# Estimate state sequence
seq = viterbi_algorithm(A, B, pi, obs)

# Print sequence
for s in seq:
    print(s)
