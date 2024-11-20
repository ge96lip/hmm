import sys
import math


def forward_pass(A, B, pi, obs):
    T = len(obs)
    N = len(A)

    alpha = [[0] * N for _ in range(T)]
    c = [0] * T

    # Initialize alpha[0]
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs[0]]
        c[0] += alpha[0][i]

    # Scale alpha[0]
    c[0] = 1 / c[0]
    for i in range(N):
        alpha[0][i] *= c[0]

    # Recursive step
    for t in range(1, T):
        c[t] = 0
        for i in range(N):
            alpha[t][i] = 0
            for j in range(N):
                alpha[t][i] += alpha[t - 1][j] * A[j][i]
            alpha[t][i] *= B[i][obs[t]]
            c[t] += alpha[t][i]

        # Scale alpha
        c[t] = 1 / c[t]
        for i in range(N):
            alpha[t][i] *= c[t]

    return alpha, c

def backward_pass(A, B, pi, obs, c):
    T = len(obs)
    N = len(A)

    beta = [[0] * N for _ in range(T)]

    # Initialize beta[T-1]
    for i in range(N):
        beta[T - 1][i] = c[T - 1]

    # Recursive step
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t][i] = 0
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]
            beta[t][i] *= c[t]

    return beta


def compute_gammas(alpha, beta, A, B, pi, obs):
    T = len(obs)
    N = len(A)

    gamma = [[0] * N for _ in range(T)]
    di_gamma = [[[0] * N for _ in range(N)] for _ in range(T - 1)]

    for t in range(T - 1):
        for i in range(N):
            gamma[t][i] = 0
            for j in range(N):
                di_gamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]
                gamma[t][i] += di_gamma[t][i][j]

    # Special case for gamma at T-1
    for i in range(N):
        gamma[T - 1][i] = alpha[T - 1][i]

    return gamma, di_gamma


def update_model(gamma, di_gamma, A, B, pi, obs):
    T = len(obs)
    N = len(A)
    M = len(B[0])

    # Re-estimate A
    for i in range(N):
        denom = 0
        for t in range(T - 1):
            denom += gamma[t][i]
        for j in range(N):
            numer = 0
            for t in range(T - 1):
                numer += di_gamma[t][i][j]
            A[i][j] = numer / denom

    # Re-estimate B
    for i in range(N):
        denom = 0
        for t in range(T):
            denom += gamma[t][i]
        for j in range(M):
            numer = 0
            for t in range(T):
                if obs[t] == j:
                    numer += gamma[t][i]
            B[i][j] = numer / denom

    # Re-estimate pi
    for i in range(N):
        pi[i] = gamma[0][i]

    return A, B, pi, obs


def baum_welch_algorithm(A, B, pi, obs):

    maxIters = 100
    iters = 0
    oldLogProb = -math.inf

    while True:

        alpha, c = forward_pass(A, B, pi, obs)
        beta = backward_pass(A, B, pi, obs, c)
        gamma, di_gamma = compute_gammas(alpha, beta, A, B, pi, obs)

        # Re-estimate lambda
        update_model(gamma, di_gamma, A, B, pi, obs)

        # Compute log-probability
        T = len(obs)
        logProb = 0
        for i in range(T):
            logProb += math.log(c[i])
        logProb = -logProb

        # Check convergence criterion
        iters += 1
        #print(logProb)
        if iters >= maxIters or logProb <= oldLogProb:
            return A, B
        oldLogProb = logProb


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

# Estimate model parameters
A, B = baum_welch_algorithm(A, B, pi, obs)

# Format the result
A_out = f"{len(A)} {len(A[0])} "
A_out += ' '.join([str(round(value, 6)) for row in A for value in row])
print(A_out)

B_out = f"{len(B)} {len(B[0])} "
B_out += ' '.join([str(round(value, 6)) for row in B for value in row])
print(B_out)
