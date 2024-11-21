#!/usr/bin/env python3

import math
from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random


class HMM:
    def __init__(self, n_species, n_obs):
        self.N = n_species
        self.M = n_obs

        # Initialize model parameters
        self.A = [[100 + random.random() for _ in range(n_species)] for _ in range(n_species)]
        self.B = [[100 + random.random() for _ in range(n_obs)] for _ in range(n_species)]
        self.pi = [100 + random.random() for _ in range(n_species)]

        # Normalize probabilities
        self._normalize_matrix(self.A)
        self._normalize_matrix(self.B)
        self._normalize_vector(self.pi)
        
    def set_A(self, A):
        self.A = A

    def set_B(self, B):
        self.B = B

    def set_PI(self, pi):
        self.pi = pi
        
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
        N = len(self.A)

        alpha = [[0] * N for _ in range(T)]
        
        # alpha_1[i] = b_i*(o_1)*π_i
        # loop over all possible states 
        for i in range(N):
            # initial state * observation probability given first observation
            alpha[0][i] = self.pi[i] * self.B[i][obs[0]]
            
        # at t we need to marginalize over the probability of having been in any other state at t-1 * matching observation probability
        for t in range(1, T):
            # loop over all possible states 
            for j in range(N):
                # b_i*(o_t)*sum[a_{j,i}*α_{t−1}(j)]
                alpha[t][j] = sum(alpha[t - 1][i] * self.A[i][j] for i in range(N)) * self.B[j][obs[t]]
                
        # compute the probability of having observed the final output sequence 
        return sum(alpha[T-1])

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
    
    def forward_pass(self, A, B, pi, obs):
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

    def backward_pass(self, A, B, pi, obs, c):
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
        N = len(self.A)
        M = len(self.B[0])

        # Re-estimate A
        for i in range(N):
            denom = 0
            for t in range(T - 1):
                denom += gamma[t][i]
            for j in range(N):
                numer = 0
                for t in range(T - 1):
                    numer += di_gamma[t][i][j]
                self.A[i][j] = numer / denom

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
                self.B[i][j] = numer / denom
                
    def calculate_params(self, A, B, pi, obs): 
        T = len(obs)
        
        maxIters = 5
        iters = 0
        oldLogProb = -math.inf
        logProb = 1
        
        while True:
            
            # alpha pass: 
            alpha, c = self.forward_pass(A, B, pi, obs)
            beta = self.backward_pass(A, B, pi, obs, c)
            gamma, di_gamma = self.compute_gammas(obs, alpha, beta)
            
            self.update_model(obs, gamma, di_gamma)
            
            # Compute log-probability
            T = len(obs)
            logProb = 0
            for i in range(T):
                logProb += math.log(c[i])
            logProb = -logProb

            # Check convergence criterion
            iters += 1
            if iters >= maxIters or logProb <= oldLogProb:
                return A, B, pi
            
            oldLogProb = logProb
        

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    
    def init_parameters(self):
        self.hmm_models = [HMM(1, 8) for _ in range(N_SPECIES)]
        self.observations = {}  # Store sequences for each fish
        self.revealed_fish = {}  # Track fish already revealed
        self.guess_threshold = 0.19  # Threshold to make a confident guess
        # for each fish make a list of observations 
        self.fishes = [(i, []) for i in range(N_FISH)]
        
    def train(self, fish_id):
        model = self.hmm_models[fish_id]
        A, B, pi = model.calculate_params(model.A, model.B, model.pi, self.obs)
        self.hmm_models[fish_id].set_A(A)
        self.hmm_models[fish_id].set_B(B)
        self.hmm_models[fish_id].set_PI(pi)
    

    def guess(self, step, observations):
        """
        This method is called every iteration with new observations -> giving new information. 
        The player OPTIONALLY makes a guess. If they choose to do so, a tuple is returned containing fish index and guess
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        # store observation for each fish if one is made 
        for i in range(len(self.fishes)):
            self.fishes[i][1].append(observations[i])
        
        """# Add movement for each fish
        for fish_id, movement in enumerate(observations):
            if fish_id not in self.observations:
                self.observations[fish_id] = []  # Initialize sequence for the fish
            self.observations[fish_id].append(movement)"""
           
        # collect observations:  
        if step < 70:      # 110 = 180 timesteps - 70 guesses
            return None
        # make a guess
        else: 
            fish_id, obs = self.fishes.pop()
            fish_type = 0 
            max_probability = 0
            # Compute probabilities for each species
            for model, species in zip(self.hmm_models, range(N_SPECIES)):
                alpha = model.forward(obs)
                # print(alpha)
                prob = alpha # sum(alpha[species])
                if prob > max_probability:
                    max_probability = prob
                    fish_type = species
            self.obs = obs
            return fish_id, fish_type
            

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

        if not correct: 
            self.train(true_type)
