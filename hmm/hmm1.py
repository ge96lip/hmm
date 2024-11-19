import sys

def forward_pass(A, B, pi, obs):
    T = len(obs)
    N = len(A)

    alpha = [[0] * N for _ in range(T)]
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = sum(alpha[t - 1][i] * A[i][j] for i in range(N)) * B[j][obs[t]]

    return sum(alpha[T-1])

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

# Calculate probability of observation sequence
prob = forward_pass(A, B, pi, obs)
print(round(prob, 6))
