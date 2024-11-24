# Define Baum-Welch algorithm
import math
import random
import sys

import numpy as np

# Define target matrices for comparison
target_A = [
    [0.7, 0.05, 0.25],
    [0.1, 0.8, 0.1],
    [0.2, 0.3, 0.5]
]

target_B = [
    [0.7, 0.2, 0.1, 0],
    [0.1, 0.4, 0.3, 0.2],
    [0, 0.1, 0.2, 0.7]
]

target_pi = [1, 0, 0]

def calculate_mse(matrix1, matrix2):
    """Calculate Mean Squared Error between two matrices."""
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    return np.mean((matrix1 - matrix2) ** 2)


def compare_matrices(A, B, pi, target_A, target_B, target_pi):
    """Compare estimated matrices with target matrices and print MSE."""
    mse_A = calculate_mse(A, target_A)
    mse_B = calculate_mse(B, target_B)
    mse_pi = calculate_mse(pi, target_pi)

    print(f"Mean Squared Error for A: {mse_A}")
    print(f"Mean Squared Error for B: {mse_B}")
    print(f"Mean Squared Error for pi: {mse_pi}")

def round_matrix(matrix, decimals=2):
    """Round all elements in a matrix or list to the specified number of decimals."""
    if isinstance(matrix[0], list):  # Matrix
        return [[round(value, decimals) for value in row] for row in matrix]
    else:  # Vector
        return [round(value, decimals) for value in matrix]
    
def baum_welch_algorithm_with_convergence(A, B, pi, obs, max_iters=100, epsilon=1e-6):
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

    def compute_gammas(alpha, beta, A, B, obs):
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
            denom = sum(gamma[t][i] for t in range(T - 1))
            for j in range(N):
                numer = sum(di_gamma[t][i][j] for t in range(T - 1))
                A[i][j] = numer / denom if denom != 0 else 0

        # Re-estimate B
        for i in range(N):
            denom = sum(gamma[t][i] for t in range(T))
            for j in range(M):
                numer = sum(gamma[t][i] for t in range(T) if obs[t] == j)
                B[i][j] = numer / denom if denom != 0 else 0

        # Re-estimate pi
        for i in range(N):
            pi[i] = gamma[0][i]

        return A, B, pi

    old_log_prob = -math.inf
    for iteration in range(max_iters):
        alpha, c = forward_pass(A, B, pi, obs)
        beta = backward_pass(A, B, pi, obs, c)
        gamma, di_gamma = compute_gammas(alpha, beta, A, B, obs)
        A, B, pi = update_model(gamma, di_gamma, A, B, pi, obs)

        # Compute log-probability
        log_prob = -sum(math.log(c_t) for c_t in c)

        # Check for convergence
        if abs(log_prob - old_log_prob) < epsilon:
            A = round_matrix(A)
            B = round_matrix(B)
            pi = round_matrix(pi)
            return iteration + 1, A, B, pi, True  # Return iteration count and final model
        old_log_prob = log_prob
    A = round_matrix(A)
    B = round_matrix(B)
    pi = round_matrix(pi)
    return max_iters, A, B, pi, False

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

max_observations = 10000
initial_obs = 100
experiment_runs = 1
total_iterations = 0
total_observations = initial_obs
successful_runs = 0
step = 10  # Increment observation size in steps
random.seed(42)

for experiment in range(experiment_runs):
    print("Experiement: ", experiment)
    for num_obs in range(0, max_observations + 1, step):
        print(f"Running with {initial_obs + num_obs} observation.")
        epsilon = max(1/math.sqrt(initial_obs + num_obs), 1e-6)
        print("Epsilon is: ", epsilon)
        obs_subset = obs[:(initial_obs + num_obs)]
        iterations, A_return, B_return, pi_return, converged = baum_welch_algorithm_with_convergence(A, B, pi, obs_subset, epsilon=epsilon)
        if converged:
            print(f"Algorithm converged with {num_obs} observations. For experiement run {experiment}")
            print(f"Converged in {iterations} iterations.")
            #print(f"Estimated A: {A_return}")
            #print(f"Estimated B: {B_return}")
            #print(f"Estimated pi: {pi_return}")

            # Compare matrices
            compare_matrices(A_return, B_return, pi_return, target_A, target_B, target_pi)
            total_iterations += iterations
            total_observations += num_obs
            successful_runs += 1
            print(f"Converged in {iterations} iterations.")
            break
        
# Calculate averages
average_iterations = total_iterations / successful_runs if successful_runs > 0 else 0
average_observations = total_observations / successful_runs if successful_runs > 0 else 0

print(f"Average iterations to converge: {average_iterations}")
print(f"Average observations to converge: {average_observations}")
print(f"Total successful runs: {successful_runs} out of {experiment_runs}")

# print(f"A is: \n{A}, \n B is: \n {B} \npi is: \n {pi}")
