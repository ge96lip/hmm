#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import sys


class HMM:
    def __init__(self, n_states, n_obs):
        self.N = n_states
        self.M = n_obs

        # Initialize model parameters
        self.A = [[100 + random.random() for _ in range(n_states)] for _ in range(n_states)]
        self.B = [[100 + random.random() for _ in range(n_obs)] for _ in range(n_states)]
        self.pi = [100 + random.random() for _ in range(n_states)]

        # Normalize probabilities
        self._normalize_matrix(self.A)
        self._normalize_matrix(self.B)
        self._normalize_vector(self.pi)

    def _normalize_matrix(self, matrix):
        for row in matrix:
            row_sum = sum(row)
            for i in range(len(row)):
                row[i] /= row_sum

    def _normalize_vector(self, vector):
        total = sum(vector)
        for i in range(len(vector)):
            vector[i] /= total


    def forward(self, obs):
        T = len(obs)

        alpha = [[0] * self.N for _ in range(T)]
        alpha_scaling = [0] * T

        # Initialize alpha[0]
        for i in range(self.N):
            alpha[0][i] = self.pi[i] * self.B[i][obs[0]]
        alpha_scaling[0] = 1 / sum(alpha[0])
        for i in range(self.N):
            alpha[0][i] *= alpha_scaling[0]

        # Recursive step
        for t in range(1, T):
            for j in range(self.N):
                alpha[t][j] = sum(alpha[t - 1][i] * self.A[i][j] for i in range(self.N)) * self.B[j][obs[t]]
            alpha_scaling[t] = 1 / sum(alpha[t])
            for j in range(self.N):
                alpha[t][j] *= alpha_scaling[t]

        return alpha, alpha_scaling

    def backward(self, obs):
        T = len(obs)

        beta = [[0] * self.N for _ in range(T)]
        beta_scaling = [0] * T

        # Initialize beta[T-1]
        for i in range(self.N):
            beta[T - 1][i] = 1
        beta_scaling[T - 1] = 1 / sum(beta[T - 1])
        for i in range(self.N):
            beta[T - 1][i] *= beta_scaling[T - 1]

        # Recursive step
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                beta[t][i] = sum(self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j] for j in range(self.N))
            beta_scaling[t] = 1 / sum(beta[t])
            for j in range(self.N):
                beta[t][j] *= beta_scaling[t]

        return beta

    def compute_gammas(self, obs, alpha, beta):
        T = len(obs)

        gamma = [[0] * self.N for _ in range(T)]
        di_gamma = [[[0] * self.N for _ in range(self.N)] for _ in range(T - 1)]

        for t in range(T - 1):
            denom = sum(
                alpha[t][i] * self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j]
                for i in range(self.N) for j in range(self.N)
            )

            for i in range(self.N):
                for j in range(self.N):
                    contribution = alpha[t][i] * self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j] / denom
                    gamma[t][i] += contribution
                    di_gamma[t][i][j] = contribution

        # Special case for gamma at T-1
        denom = sum(alpha[T - 1][i] for i in range(self.N))
        for i in range(self.N):
            gamma[T - 1][i] = alpha[T - 1][i] / denom

        return gamma, di_gamma

    def update_model(self, obs, gamma, di_gamma):
        T = len(obs)
        epsilon = 1e-6  # Small constant to prevent zeros

        # Re-estimate A
        for i in range(self.N):
            denom = sum(gamma[t][i] for t in range(T - 1)) + epsilon * self.N
            for j in range(self.N):
                numer = sum(di_gamma[t][i][j] for t in range(T - 1)) + epsilon
                self.A[i][j] = numer / denom

        # Re-estimate B
        for i in range(self.N):
            denom = sum(gamma[t][i] for t in range(T)) + epsilon * self.M
            for k in range(self.M):
                numer = sum(gamma[t][i] for t in range(T) if obs[t] == k) + epsilon
                self.B[i][k] = numer / denom

        # Re-estimate pi
        for i in range(self.N):
            self.pi[i] = gamma[0][i] + epsilon
        self._normalize_vector(self.pi)

    def train(self, obs):

        while True:

            # Step 2: Compute all values
            alpha, alpha_scaling = self.forward(obs)
            beta = self.backward(obs)
            gamma, di_gamma = self.compute_gammas(obs, alpha, beta)

            # Step 3: Re-estimate lambda
            prev_A = [row[:] for row in self.A]
            prev_B = [row[:] for row in self.B]
            self.update_model(obs, gamma, di_gamma)

            # Step 4: Repeat until convergence
            max_change_A = max(abs(self.A[i][j] - prev_A[i][j]) for i in range(self.N) for j in range(self.N))
            max_change_B = max(abs(self.B[i][j] - prev_B[i][j]) for i in range(self.N) for j in range(self.M))
            threshold = 1e-5
            if max_change_A < threshold or max_change_B < threshold:
                break



class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        self.hmm_models = [HMM(5, 8) for _ in range(N_SPECIES)]
        self.observations = {}  # Store sequences for each fish
        self.revealed_fish = {}  # Track fish already revealed
        self.guess_threshold = 0.19  # Threshold to make a confident guess

    def guess(self, step, observations):
        # Add movement for each fish
        for fish_id, movement in enumerate(observations):
            if fish_id not in self.observations:
                self.observations[fish_id] = []  # Initialize sequence for the fish
            self.observations[fish_id].append(movement)

        # Store likelihoods for all unrevealed fish
        fish_likelihoods = []

        # Compute likelihoods for all unrevealed fish
        for fish_id in range(len(observations)):
            # Skip already revealed fish
            if fish_id in self.revealed_fish:
                continue

            sequence = self.observations[fish_id]

            # Evaluate likelihood of sequence for each species
            species_likelihoods = []
            for species in range(N_SPECIES):
                alpha, alpha_scaling = self.hmm_models[species].forward(sequence)
                if sum(alpha[-1]) == 0 or alpha_scaling[-1] == 0:
                    likelihood = 0
                else:
                    likelihood = sum(alpha[-1]) / alpha_scaling[-1]
                species_likelihoods.append(likelihood)

            # Get the species with the highest likelihood
            best_species = species_likelihoods.index(max(species_likelihoods))
            best_likelihood = max(species_likelihoods)

            # Store the likelihood along with fish_id and species
            fish_likelihoods.append((best_likelihood, -len(sequence), fish_id, best_species))

        # If we have any fish to guess
        if fish_likelihoods:
            # Sort fish by likelihood (descending), then by sequence length (descending)
            fish_likelihoods.sort(reverse=True)

            # Get the fish with the highest likelihood
            best_likelihood, _, best_fish_id, best_species = fish_likelihoods[0]
            print(best_likelihood)

            if True:#best_likelihood > self.guess_threshold:
                self.revealed_fish[best_fish_id] = best_species

                return best_fish_id, best_species

        return None

    def reveal(self, correct, fish_id, true_type):
        """
        This method is called whenever a guess is made.
        It updates the HMM corresponding to the true species of the fish.
        :param correct: True if the guess was correct, False otherwise
        :param fish_id: Index of the fish that was guessed
        :param true_type: The correct type of the fish
        """
        # Map fish_id to true_type
        self.revealed_fish[fish_id] = true_type

        # Update the HMM for the true species
        self.hmm_models[true_type].train(self.observations[fish_id])
