# Define Baum-Welch algorithm
import math
import random
import sys
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import time
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

given_A = [
    [0.54, 0.26, 0.20],
    [0.19, 0.53, 0.28],
    [0.22, 0.18, 0.6]
]

given_B = [
    [0.5, 0.2, 0.11, 0.19],
    [0.22, 0.28, 0.23, 0.27 ],
    [0.19, 0.21, 0.15, 0.45]
]

given_pi = [0.3, 0.2, 0.5]
def permute_matrices(A, B, pi, perm):
    """Permute rows/columns of A, B, and pi based on the given permutation."""
    A_perm = np.array(A)[perm, :][:, perm]
    B_perm = np.array(B)[perm, :]
    pi_perm = np.array([pi[i] for i in perm])
    return A_perm.tolist(), B_perm.tolist(), pi_perm.tolist()

def find_closest_match(A, B, pi, target_A, target_B, target_pi):
    """Find the permutation that minimizes the MSE."""
    n = len(pi)
    perms = list(permutations(range(n)))
    best_mse = float("inf")
    best_perm = None
    best_A, best_B, best_pi = None, None, None

    for perm in perms:
        A_perm, B_perm, pi_perm = permute_matrices(A, B, pi, perm)
        mse_A = calculate_mse(A_perm, target_A)
        mse_B = calculate_mse(B_perm, target_B)
        mse_pi = calculate_mse(pi_perm, target_pi)
        total_mse = mse_A + mse_B + mse_pi

        if total_mse < best_mse:
            best_mse = total_mse
            best_perm = perm
            best_A, best_B, best_pi = A_perm, B_perm, pi_perm

    return mse_A, mse_B, mse_pi, best_A, best_B, best_pi

# Initialize HMM parameters
def initialize_hmm_params(strategy, obs, N=3, M=4):
    """Initialize HMM parameters based on the chosen strategy."""
    
    if strategy == "uniform":
        # Uniform initialization
        A = np.ones((N, N)) / N
        B = np.ones((N, M)) / M
        pi = np.ones(N) / N
    elif strategy == "original": 
        A = np.array(given_A)
        B = np.array(given_B)
        pi = np.array(given_pi)
    elif strategy == "frequency_based":

        # Frequency-based initialization (dummy example with uniform here; replace with real counts if available)
        state_mapped_obs = [o % N for o in obs]  # Map to states using modulo N
        emission_mapped_obs = [o % M for o in obs]  # Map to emissions using modulo M

        # Initialize frequency counts
        transition_counts = np.zeros((N, N))
        emission_counts = np.zeros((N, M))
        initial_counts = np.zeros(N)

        # Compute frequencies for transitions, emissions, and initial states
        initial_counts[state_mapped_obs[0]] += 1  # Count the initial state
        for t in range(len(state_mapped_obs) - 1):
            current_state = state_mapped_obs[t]
            next_state = state_mapped_obs[t + 1]
            emission_counts[current_state, emission_mapped_obs[t]] += 1  # Count emissions
            transition_counts[current_state, next_state] += 1  # Count transitions

        # Include the final emission
        emission_counts[state_mapped_obs[-1], emission_mapped_obs[-1]] += 1

        # Normalize to convert counts to probabilities
        A = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        
        B = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        
        pi = initial_counts / initial_counts.sum()
        
        # Handle potential NaN values due to zero divisions
        A = np.nan_to_num(A)
        B = np.nan_to_num(B)
        pi = np.nan_to_num(pi)
    
        

    elif strategy == "perturbed_target":
        # Perturbed initialization near the target matrices
        A = np.array(target_A) + np.random.normal(0, 0.01, size=(N, N))
        B = np.array(target_B) + np.random.normal(0, 0.01, size=(N, M))
        pi = np.array(target_pi) + np.random.normal(0, 0.01, size=(N,))

        # Ensure non-negative values
        A = np.clip(A, 0, None)
        B = np.clip(B, 0, None)
        pi = np.clip(pi, 0, None)

        
    elif strategy == "random": 
        A = np.random.rand(N, N)
        B = np.random.rand(N, M)
        pi = np.random.rand(N)
    else:
        raise ValueError("Invalid initialization strategy")
    
    # Normalize to ensure valid probability distributions
    A = normalize_matrix(A.tolist())
    B = normalize_matrix(B.tolist())
    pi = normalize_vector(pi.tolist())

    return A, B, pi

def calculate_mse(matrix1, matrix2):
    """Calculate Mean Squared Error between two matrices."""
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    return np.mean((matrix1 - matrix2) ** 2)
def calculate_mse_with_truncation(matrix1, matrix2):
    # Check if the inputs are 1D vectors
    if isinstance(matrix1[0], (float, int)) and isinstance(matrix2[0], (float, int)):
        # Truncate to the shorter length for vectors
        min_length = min(len(matrix1), len(matrix2))
        truncated_matrix1 = matrix1[:min_length]
        truncated_matrix2 = matrix2[:min_length]
        # Compute MSE
        return np.mean((np.array(truncated_matrix1) - np.array(truncated_matrix2)) ** 2)
    else:
        # Determine the common dimensions for matrices
        min_rows = min(len(matrix1), len(matrix2))
        min_cols = min(len(matrix1[0]), len(matrix2[0]))
        
        # Truncate both matrices
        truncated_matrix1 = [row[:min_cols] for row in matrix1[:min_rows]]
        truncated_matrix2 = [row[:min_cols] for row in matrix2[:min_rows]]
        
        # Compute MSE
        return np.mean((np.array(truncated_matrix1) - np.array(truncated_matrix2)) ** 2)
def calculate_mse_with_padding(matrix1, matrix2):
    # Check if the inputs are 1D vectors
    if isinstance(matrix1[0], (float, int)) and isinstance(matrix2[0], (float, int)):
        # Pad vectors to the same length
        max_length = max(len(matrix1), len(matrix2))
        padded_matrix1 = np.pad(matrix1, (0, max_length - len(matrix1)), constant_values=0)
        padded_matrix2 = np.pad(matrix2, (0, max_length - len(matrix2)), constant_values=0)
        # Compute MSE
        return np.mean((np.array(padded_matrix1) - np.array(padded_matrix2)) ** 2)
    else:
        # Determine the maximum dimensions for matrices
        max_rows = max(len(matrix1), len(matrix2))
        max_cols = max(len(matrix1[0]), len(matrix2[0]))
        
        # Pad both matrices
        padded_matrix1 = [
            row + [0] * (max_cols - len(row)) for row in matrix1
        ] + [[0] * max_cols] * (max_rows - len(matrix1))
        padded_matrix2 = [
            row + [0] * (max_cols - len(row)) for row in matrix2
        ] + [[0] * max_cols] * (max_rows - len(matrix2))
        
        # Compute MSE
        return np.mean((np.array(padded_matrix1) - np.array(padded_matrix2)) ** 2)

def compare_matrices(A, B, pi, target_A, target_B, target_pi):
    """Compare estimated matrices with target matrices and print MSE."""
    mse_A = calculate_mse(A, target_A)
    mse_B = calculate_mse(B, target_B)
    mse_pi = calculate_mse(pi, target_pi)

    print(f"Mean Squared Error for A: {mse_A}")
    print(f"Mean Squared Error for B: {mse_B}")
    print(f"Mean Squared Error for pi: {mse_pi}")
    return mse_A, mse_B, mse_pi

def round_matrix(matrix, decimals=2):
    """Round all elements in a matrix or list to the specified number of decimals."""
    if isinstance(matrix[0], list):  # Matrix
        return [[round(value, decimals) for value in row] for row in matrix]
    else:  # Vector
        return [round(value, decimals) for value in matrix]
def map_observations(obs, M):
    """Map observations to the range [0, M-1]."""
    return [o % M for o in obs]
   
def baum_welch_algorithm_with_convergence(A_baum, B_baum, pi_baum, obs, max_iters=100, epsilon=1e-6):
    
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
            denom = 0
            for t in range(T - 1):
                denom += gamma[t][i]
            for j in range(N):
                numer = 0
                for t in range(T - 1):
                    numer += di_gamma[t][i][j]
                A[i][j] = numer / (denom + epsilon)

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
                B[i][j] = numer / (denom + epsilon)
        return A, B, pi

    old_log_prob = -math.inf
    log_likelihoods = []
    for iteration in range(max_iters):
        alpha, c = forward_pass(A_baum, B_baum, pi_baum, obs)
        beta = backward_pass(A_baum, B_baum, pi_baum, obs, c)
        gamma, di_gamma = compute_gammas(alpha, beta, A_baum, B_baum, obs)
        A_baum, B_baum, pi_baum = update_model(gamma, di_gamma, A_baum, B_baum, pi_baum, obs)

        # Compute log-probability
        log_prob = -sum(math.log(max(c_t, 1e-10)) for c_t in c)  # Clamp c_t to avoid log(0)
        #log_prob = -sum(math.log(c_t) for c_t in c)
        log_likelihoods.append(log_prob)

        # Check for convergence
        if abs(log_prob - old_log_prob) < epsilon:
            A_baum = round_matrix(A_baum)
            B_baum = round_matrix(B_baum)
            pi_baum = round_matrix(pi_baum)
            return iteration + 1, A_baum, B_baum, pi_baum, True, log_likelihoods  # Return iteration count and final model
        old_log_prob = log_prob
    A_baum = round_matrix(A_baum)
    B_baum = round_matrix(B_baum)
    pi_baum = round_matrix(pi_baum)
    return max_iters, A_baum, B_baum, pi_baum, False, log_likelihoods

def evaluate_model_log_likelihood(obs, A, B, pi):
    """Calculate log-likelihood of observations given the model."""
    alpha, _ = forward_pass(A, B, pi, obs)  # Use forward algorithm
    log_likelihood = sum([math.log(c_t) for c_t in alpha])
    return log_likelihood

def convert_to_matrix(data, num_rows, num_cols):
    data = list(map(float, data))
    return [data[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

def normalize_matrix(matrix):
    for row in matrix:
        row_sum = sum(row)
        if row_sum > 0:  # Only normalize if the row sum is non-zero
            for i in range(len(row)):
                row[i] /= row_sum
    return matrix

def normalize_vector(vector):
    total = sum(vector)
    for i in range(len(vector)):
        vector[i] /= total
    return vector

def question_7(obs): 
    # Read input
    A, B, pi = initialize_hmm_params("original", obs)
    run_experiments(A, B, pi, obs, experiment_repetitions=1)
    
    
def question_8(obs): 
    
    possible_initalization_methods = ["uniform", "frequency_based", "random", "perturbed_target", "original"]
    # uniform = 0.34023055555555554 Best iteration: 3, Best observation number: 7100
    # frequency = 0.2534611111111111 Best iteration: 3, Best observation number: 1000
    # random = 0.356 Best iteration: 2, Best observation number: 11000
    
    overall_mse = np.inf
    best_A, best_B, best_pi = None, None, None
    best_strategy = ""
    for i, strategy in enumerate(possible_initalization_methods): 
        print("Running strategy: ", strategy)
        A, B, pi = initialize_hmm_params(strategy, obs)
        

        return_A, return_B, return_pi, total_mse = run_experiments(A, B, pi, obs)
        #print(total_mse, overall_mse)
        if total_mse < overall_mse: 
            
            best_A = return_A
            best_B = return_B
            best_pi = return_pi
            best_strategy = strategy
            overall_mse = total_mse
            
    return overall_mse, best_strategy, best_A, best_B, best_pi

def run_experiments(A, B, pi, obs, max_observations = 1000, initial_obs = 1000, experiment_repetitions = 1, step_size=100): 

    best_mse_A = float("inf")
    best_mse_B = float("inf")
    best_mse_pi = float("inf")
    best_A, best_B, best_pi = None, None, None
    best_iterations = 0
    best_observations = 0

    total_iterations = 0
    total_observations = initial_obs
    successful_runs = 0
    best_mse = np.inf
    for experiment in range(experiment_repetitions):
        print("Experiement: ", experiment)
        for num_obs in range(0, max_observations + 1, step_size):
            current_obs = (initial_obs + num_obs)
            print(f"Running with {current_obs} observation.")
            epsilon = max(1/math.sqrt(current_obs), 1e-6) # taking max yields better results than taking min 
            #print("Epsilon is: ", epsilon)
            obs_subset = obs[:(current_obs)]
            iterations, A_return, B_return, pi_return, converged, log = baum_welch_algorithm_with_convergence(A, B, pi, obs_subset, epsilon=epsilon)
            if converged:
                print(f"Algorithm converged with {current_obs} observations. For experiement run {experiment}")
                print(f"Converged in {iterations} iterations.")
                # Compare matrices
                
                mse_A, mse_B, mse_pi, perm_A, perm_B, perm_pi = find_closest_match(A_return, B_return, pi_return, target_A, target_B, target_pi)
        
                # mse_A, mse_B, mse_pi = compare_matrices(A_return, B_return, pi_return, target_A, target_B, target_pi)
                total_mse = mse_A + mse_B + mse_pi
                if total_mse < best_mse:
                    print(f"MSE improved for {current_obs} observations.")
                    best_mse_A, best_mse_B, best_mse_pi = mse_A, mse_B, mse_pi
                    best_A, best_B, best_pi = perm_A, perm_B, perm_pi
                    best_mse = total_mse
                    best_iterations = iterations
                    best_observations = current_obs

                else: 
                    print(f"MSE did not improve for any matrix for {current_obs} observations.")
                    #break 
                    
                total_iterations += iterations
                total_observations += num_obs
                successful_runs += 1
                #print(f"Converged in {iterations} iterations.")
                
            
    # Calculate averages
    average_iterations = total_iterations / successful_runs if successful_runs > 0 else 0
    average_observations = total_observations / successful_runs if successful_runs > 0 else 0
    print("Number of total observation: ", len(obs))
    print(f"{best_iterations} iterations until convergence with run with highest MSE")
    print(f"Best run had {best_observations} observations for training")
    print(f"Best MSE for A: {best_mse_A}")
    print(f"Best MSE for B: {best_mse_B}")
    print(f"Best MSE for pi: {best_mse_pi}")
    print(f"Best total MSE: ", best_mse)
    #print(f"Average iterations to converge: {average_iterations}")
    #print(f"Average observations to converge: {average_observations}")
    #print(f"Total successful runs: {successful_runs} out of {experiment_repetitions}")
    
    return best_A, best_B, best_pi, best_mse


def question_9(obs, target_A, target_B, target_pi, max_states=5, max_emissions=6):
    results = []
    for N in range(2, max_states + 1):  # Test varying number of states
        for M in range(2, max_emissions + 1):  # Test varying number of emissions
            print(f"Testing with {N} states and {M} emissions...")
            A, B, pi = initialize_hmm_params("frequency_based", obs, N, M)
            # get best permutation 
            #_, _, _, A, B, pi = find_closest_match(A, B, pi, target_A, target_B, target_pi)
            
            epsilon = max(1/math.sqrt(len(obs)), 1e-6) # taking max yields better results than taking min 
            # Train the HMM using Baum-Welch
            mapped_obs = map_observations(obs, len(B[0])) 
            iterations, A_return, B_return, pi_return, converged = baum_welch_algorithm_with_convergence(
                A, B, pi, mapped_obs, max_iters=100, epsilon=epsilon
            )
            
            if converged:
                print(f"Converged for {N} states and {M} emissions.")
                
                # Evaluate the model
                total_mse = np.inf
                mse_A, mse_B, mse_pi = np.inf, np.inf, np.inf
                if (
                    len(target_A) == len(A_return) and  # Check number of rows in A
                    all(len(row) == len(target_A[0]) for row in A_return) and  # Check number of columns in A
                    len(target_B) == len(B_return) and  # Check number of rows in B
                    all(len(row) == len(target_B[0]) for row in B_return) and  # Check number of columns in B
                    len(target_pi) == len(pi_return)  # Check length of pi
                ):
                    # Dimensions match, proceed with the comparison
                    mse_A, mse_B, mse_pi, perm_A, perm_B, perm_pi = find_closest_match(
                        A_return, B_return, pi_return, target_A, target_B, target_pi
                    )                   
                    total_mse = mse_A + mse_B + mse_pi
                else: 
                    """mse_A = calculate_mse_with_truncation(A_return, target_A)
                    mse_B = calculate_mse_with_truncation(B_return, target_B)
                    mse_pi = calculate_mse_with_truncation(pi_return, target_pi)"""
                    mse_A = calculate_mse_with_padding(A_return, target_A)
                    mse_B = calculate_mse_with_padding(B_return, target_B)
                    mse_pi = calculate_mse_with_padding(pi_return, target_pi)
                    
                    total_mse = mse_A + mse_B + mse_pi
                # Calculate AIC and BIC
                num_params = N * (N - 1) + N * M + (N - 1)  # Parameters: A, B, and pi
                log_likelihood = -iterations  # Placeholder for log-likelihood
                aic = 2 * num_params - 2 * log_likelihood
                bic = num_params * math.log(len(obs)) - 2 * log_likelihood
                
                # Save results
                results.append({
                    "states": N,
                    "emissions": M,
                    "mse_A": mse_A,
                    "mse_B": mse_B,
                    "mse_pi": mse_pi,
                    "total_mse": total_mse,
                    "aic": aic,
                    "bic": bic,
                    "iterations": iterations,
                })
                print(f"Results: MSE(total): {total_mse}, MSE(A): {mse_A}, MSE(B): {mse_B}, MSE(pi): {mse_pi}, AIC: {aic}, BIC: {bic}")
            else:
                print(f"Did not converge for {N} states and {M} emissions.")
    # Find the best configuration
    best_result = min(results, key=lambda x: x["bic"])  # Choose the model with the lowest BIC
    print("Best Configuration BIC:")
    print(best_result)
    best_result = min(results, key=lambda x: x["aic"])  # Choose the model with the lowest BIC
    print("Best Configuration AIC:")
    print(best_result)
    best_result = min(results, key=lambda x: x["total_mse"])  # Choose the model with the lowest BIC
    print("Best Configuration MSE:")
    print(best_result)
    return results

def question10(obs, all_obs): 
    epsilon = max(1/math.sqrt(len(obs)), 1e-6) # taking max yields better results than taking min 

    print("Number of observation used during training: ", len(obs))
    A_uniform, B_uniform, pi_uniform = initialize_hmm_params("uniform", all_obs, 3, 4)
    A_random, B_random, pi_random = initialize_hmm_params("random", all_obs, 3, 4)
    A_frequency, B_frequency, pi_frequency = initialize_hmm_params("frequency_based", all_obs)
    A_perturbed, B_perturbed, pi_perturbed = initialize_hmm_params("perturbed_target", all_obs, 3, 4)

    start_time = time.time()
    iterations_uniform, A_uni, B_uni, pi_uni, conf_uni, log_likelihoods_uniform = baum_welch_algorithm_with_convergence(A_uniform, B_uniform, pi_uniform, obs, epsilon=epsilon)
    uniform_time = time.time() - start_time

    start_time = time.time()
    iterations_random, A_random, B_random, pi_random, conf_random, log_likelihoods_random = baum_welch_algorithm_with_convergence(A_random, B_random, pi_random, obs, epsilon=epsilon)
    random_time = time.time() - start_time
    
    start_time = time.time()
    iterations_frequency, A_frequ, B_frequ, pi_frequ, conf_frequ, log_likelihoods_frequency = baum_welch_algorithm_with_convergence(A_frequency, B_frequency, pi_frequency, obs, epsilon=epsilon)
    frequency_time = time.time() - start_time
    
    start_time = time.time()
    iterations_perturbed, A_pert, B_pert, pi_pert, conf_frequ, log_likelihoods_frequency = baum_welch_algorithm_with_convergence(A_perturbed, B_perturbed, pi_perturbed, obs, epsilon=epsilon)
    perturbed_time = time.time() - start_time
    
    print(f"Time (Uniform Initialization): {uniform_time} seconds")
    print(f"Time (Random Initialization): {random_time} seconds")
    print(f"Time (Frequency Initialization): {frequency_time} seconds")
    print(f"Time (Perturbed Initialization): {perturbed_time} seconds")
    
    # MSE calculations
    mse_random_A, mse_random_B, mse_random_pi, _, _, _ = find_closest_match(
                        A_random, B_random, pi_random, target_A, target_B, target_pi) 
    total_mse_random = mse_random_A + mse_random_B + mse_random_pi
    
    mse_uniform_A, mse_uniform_B, mse_uniform_pi, _, _, _ = find_closest_match(
                        A_uni, B_uni, pi_uni, target_A, target_B, target_pi) 
    total_mse_uni = mse_uniform_A + mse_uniform_B + mse_uniform_pi 
    
    print("baum welch finds: \n A: ", A_frequ, "\n B: ",B_frequ, "\n pi: ", pi_frequ,)
    mse_frequ_A, mse_frequ_B, mse_frequ_pi, freq_A, freq_B, freq_pi = find_closest_match(
                        A_frequ, B_frequ, pi_frequ, target_A, target_B, target_pi) 
    print("after closest: ", freq_A, "B: \n", freq_B,"\n pi: ", freq_pi)
    total_mse_frequ = mse_frequ_A + mse_frequ_B + mse_frequ_pi
    
    mse_perturbed_A, mse_perturbed_B, mse_perturbed_pi, _, _, _ = find_closest_match(
                        A_perturbed, B_perturbed, pi_perturbed, target_A, target_B, target_pi) 
    total_mse_perturbed = mse_perturbed_A + mse_perturbed_B + mse_perturbed_pi
    print(f"MSE (Uniform Initialization): {total_mse_uni}")
    print(f"MSE (Random Initialization): {total_mse_random}")
    print(f"MSE (Frequency Initialization): {total_mse_frequ}")
    print(f"MSE (Perturbed Initialization): {total_mse_perturbed}")
    print(f"MSE (Frequency Initialization): {total_mse_frequ}")
    
    # Run multiple experiments
    results_uniform = [baum_welch_algorithm_with_convergence(A_uniform, B_uniform, pi_uniform, obs, epsilon=epsilon) for _ in range(10)]
    results_random = [baum_welch_algorithm_with_convergence(A_random, B_random, pi_random, obs, epsilon=epsilon) for _ in range(10)]
    results_frequency = [baum_welch_algorithm_with_convergence(A_frequency, B_frequency, pi_frequency, obs, epsilon=epsilon) for _ in range(10)]
    results_perturbed = [baum_welch_algorithm_with_convergence(A_perturbed, B_perturbed, pi_perturbed, obs, epsilon=epsilon) for _ in range(10)]

    # Calculate variance of parameters
    variance_A_uniform = np.var([result[1] for result in results_uniform], axis=0)
    variance_A_random = np.var([result[1] for result in results_random], axis=0)
    variance_A_frequency = np.var([result[1] for result in results_frequency], axis=0)
    variance_A_perturbed = np.var([result[1] for result in results_perturbed], axis=0)
    print(f"Variance in A (Uniform): {variance_A_uniform}")
    print(f"Variance in A (Random): {variance_A_random}")
    print(f"Variance in A (Frequency): {variance_A_frequency}")
    print(f"Variance in A (Perturbed): {variance_A_perturbed}")
    """
    print(sum(log_likelihoods_random))
    print(sum(log_likelihoods_uniform))
    print(sum(log_likelihoods_frequency))
    """
    print(f"Iterations until convergence (Uniform): {iterations_uniform}")
    print(f"Iterations until convergence (Random): {iterations_random}")
    print(f"Iterations until convergence (Frequency): {iterations_frequency}")
    print(f"Iterations until convergence (Perturbed): {iterations_perturbed}")
    
    return freq_A, freq_B, freq_pi
obs_data = sys.stdin.readline().split()
obs = list(map(int, obs_data[1:]))

random.seed(42)

print("Number of observation: ", len(obs))
# question 7: 
question_7(obs)

# question 8: 
overall_mse, strategy, A, B, pi = question_8(obs)
print(f"{strategy} gives the lowest MSE over all strategies with: {overall_mse}")

#print("total mse: ", total_mse)
obs_selected = obs[:(1000)]
# Run the experiment
results = question_9(obs_selected, target_A, target_B, target_pi, max_states=5, max_emissions=6)
freq_A, freq_B, freq_pi = question10(obs_selected, obs)


# print(f"A is: \n{A}, \n B is: \n {B} \npi is: \n {pi}")
