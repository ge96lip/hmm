import sys


def forward_pass(A, B, pi, obs):
    T = len(obs)
    N = len(A)

    alpha = [[0] * N for _ in range(T)]
    alpha_scaling = [0] * T

    # Initialize alpha[0]
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs[0]]
    alpha_scaling[0] = 1 / sum(alpha[0])
    for i in range(N):
        alpha[0][i] *= alpha_scaling[0]

    # Recursive step
    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = sum(alpha[t - 1][i] * A[i][j] for i in range(N)) * B[j][obs[t]]
        alpha_scaling[t] = 1 / sum(alpha[t])
        for j in range(N):
            alpha[t][j] *= alpha_scaling[t]

    return alpha

def backward_pass(A, B, pi, obs):
    T = len(obs)
    N = len(A)

    beta = [[0] * N for _ in range(T)]
    beta_scaling = [0] * T

    # Initialize beta[T-1]
    for i in range(N):
        beta[T - 1][i] = 1
    beta_scaling[T - 1] = 1 / sum(beta[T - 1])
    for i in range(N):
        beta[T - 1][i] *= beta_scaling[T - 1]

    # Recursive step
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t][i] = sum(A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j] for j in range(N))
        beta_scaling[t] = 1 / sum(beta[t])
        for j in range(N):
            beta[t][j] *= beta_scaling[t]

    return beta


def compute_gammas(alpha, beta, A, B, pi, obs):
    T = len(obs)
    N = len(A)

    gamma = [[0] * N for _ in range(T)]
    di_gamma = [[[0] * N for _ in range(N)] for _ in range(T - 1)]

    for t in range(T - 1):
        denom = sum(
            alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j]
            for i in range(N) for j in range(N)
        )

        for i in range(N):
            for j in range(N):
                contribution = alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j] / denom
                gamma[t][i] += contribution
                di_gamma[t][i][j] = contribution

    # Special case for gamma at T-1
    denom = sum(alpha[T - 1][i] for i in range(N))
    for i in range(N):
        gamma[T - 1][i] = alpha[T - 1][i] / denom

    return gamma, di_gamma

def update_model(gamma, di_gamma, A, B, pi, obs):
    T = len(obs)
    N = len(A)
    M = len(B[0])

    # Re-estimate A
    for i in range(N):
        for j in range(N):
            numer = sum(di_gamma[t][i][j] for t in range(T - 1))
            denom = sum(gamma[t][i] for t in range(T - 1))
            A[i][j] = numer / denom if denom != 0 else 0

    # Re-estimate B
    for i in range(N):
        for k in range(M):
            numer = sum(gamma[t][i] for t in range(T) if obs[t] == k)
            denom = sum(gamma[t][i] for t in range(T))
            B[i][k] = numer / denom if denom != 0 else 0

    # Re-estimate pi
    for i in range(N):
        pi[i] = gamma[0][i]


def baum_welch_algorithm(A, B, pi, obs):

    while True:

        # Step 2: Compute all values
        alpha = forward_pass(A, B, pi, obs)
        beta = backward_pass(A, B, pi, obs)
        gamma, di_gamma = compute_gammas(alpha, beta, A, B, pi, obs)

        # Step 3: Re-estimate lambda
        prev_A = [row[:] for row in A]
        prev_B = [row[:] for row in B]
        update_model(gamma, di_gamma, A, B, pi, obs)

        # Step 4: Repeat until convergence
        N = len(A)
        M = len(B[0])
        max_change_A = max(abs(A[i][j] - prev_A[i][j]) for i in range(N) for j in range(N))
        max_change_B = max(abs(B[i][j] - prev_B[i][j]) for i in range(N) for j in range(M))
        threshold = 1e-5
        if max_change_A < threshold or max_change_B < threshold:
            return A, B


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

print("Estimated Transition Matrix (A):", A)
print("Estimated Emission Matrix (B):", B)