from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math


class HMM:
    def __init__(self, n_states, n_obs):
        self.N = n_states
        self.M = n_obs

        # Initialize model parameters
        self.A = [[1000 + random.random() for _ in range(n_states)] for _ in range(n_states)]
        self.B = [[1000 + random.random() for _ in range(n_obs)] for _ in range(n_states)]
        self.pi = [1000 + random.random() for _ in range(n_states)]

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

    def evaluate(self, obs):

        T = len(obs)
        N = len(self.A)

        alpha = [[0] * N for _ in range(T)]

        # Initialize alpha[0]
        for i in range(N):
            alpha[0][i] = self.pi[i] * self.B[i][obs[0]]

        # Recursive step
        for t in range(1, T):
            for i in range(N):
                alpha[t][i] = 0
                for j in range(N):
                    alpha[t][i] += alpha[t - 1][j] * self.A[j][i]
                alpha[t][i] *= self.B[i][obs[t]]

        return sum(alpha[T - 1])

    def forward(self, obs):
        T = len(obs)
        N = len(self.A)

        alpha = [[0] * N for _ in range(T)]
        c = [0] * T

        # Initialize alpha[0]
        for i in range(N):
            alpha[0][i] = self.pi[i] * self.B[i][obs[0]]
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
                    alpha[t][i] += alpha[t - 1][j] * self.A[j][i]
                alpha[t][i] *= self.B[i][obs[t]]
                c[t] += alpha[t][i]

            # Scale alpha
            c[t] = 1 / c[t]
            for i in range(N):
                alpha[t][i] *= c[t]

        return alpha, c

    def backward(self, obs, c):
        T = len(obs)
        N = len(self.A)

        beta = [[0] * N for _ in range(T)]

        # Initialize beta[T-1]
        for i in range(N):
            beta[T - 1][i] = c[T - 1]

        # Recursive step
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t][i] = 0
                for j in range(N):
                    beta[t][i] += self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j]
                beta[t][i] *= c[t]

        return beta

    def compute_gammas(self, obs, alpha, beta):
        T = len(obs)
        N = len(self.A)

        gamma = [[0] * N for _ in range(T)]
        di_gamma = [[[0] * N for _ in range(N)] for _ in range(T - 1)]

        for t in range(T - 1):
            for i in range(N):
                gamma[t][i] = 0
                for j in range(N):
                    di_gamma[t][i][j] = alpha[t][i] * self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j]
                    gamma[t][i] += di_gamma[t][i][j]

        # Special case for gamma at T-1
        for i in range(N):
            gamma[T - 1][i] = alpha[T - 1][i]

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

        # Normalize
        self._normalize_matrix(self.A)
        self._normalize_matrix(self.B)
        self._normalize_vector(self.pi)

    def train(self, obs):

        maxIters = 100
        oldLogProb = -math.inf

        for iters in range(maxIters):

            alpha, c = self.forward(obs)
            beta = self.backward(obs, c)
            gamma, di_gamma = self.compute_gammas(obs, alpha, beta)

            # Re-estimate lambda
            self.update_model(obs, gamma, di_gamma)

            # Compute log-probability
            T = len(obs)
            logProb = 0
            for i in range(T):
                logProb += math.log(c[i])
            logProb = -logProb

            # Check convergence criterion
            iters += 1
            # print(logProb)
            if logProb <= oldLogProb:
                break
            oldLogProb = logProb


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        # for each species have a HMM which has n observation for itself
        # we only have one state that matters to us e.g. swimming -> the fish movement is modeled through B
        self.hmm_models = [HMM(1, 8) for _ in range(N_SPECIES)]

        # for each fish make a list of observations
        self.fishes = [(i, []) for i in range(N_FISH)]

        self.curr_obs = None

    def guess(self, step, observations):

        # Store movement for each fish
        for i in range(len(self.fishes)):
            self.fishes[i][1].append(observations[i])

        # Start guessing after 100 steps
        if step < 100:
            return None

        fish_id, obs = self.fishes.pop()
        fish_type = 0
        max_probability = 0
        # Compute probabilities for each species
        for model, species in zip(self.hmm_models, range(N_SPECIES)):
            prob = model.evaluate(obs)
            if prob > max_probability:
                max_probability = prob
                fish_type = species
        self.curr_obs = obs
        return fish_id, fish_type

    def reveal(self, correct, fish_id, true_type):
        """
        This method is called whenever a guess is made.
        It updates the HMM corresponding to the true species of the fish.
        :param correct: True if the guess was correct, False otherwise
        :param fish_id: Index of the fish that was guessed
        :param true_type: The correct type of the fish
        """
        # Update the HMM for the true species
        if not correct:
            model = self.hmm_models[true_type]
            model.train(self.curr_obs)
