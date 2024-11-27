from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math


class HMM:
    """
    Implements a Hidden Markov Model with methods for evaluation, training, and updating model parameters.

    Attributes:
        N (int): The number of states in the HMM.
        M (int): The number of possible observations (emissions) in the HMM.
        A (list): State transition probability matrix.
        B (list): Emission probability matrix.
        pi (list): Initial state distribution.
    """
    
    def __init__(self, n_states, n_obs):
        """
        Initializes an HMM with the given number of states and observations.

        Input:
            n_states (int): The number of states in the HMM.
            n_obs (int): The number of possible observation symbols.

        Output:
            None
            - Initializes the transition matrix (A), emission matrix (B), and initial state probabilities (pi) with random values.
            - Normalizes these probabilities to ensure they sum to 1.
        """
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
        """
        Computes the probability of an observation sequence given the model using the forward algorithm.

        Input:
            obs (list): A sequence of observations (list of integers).

        Output:
            float: The probability of the observation sequence.
        """
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
        """
        Computes the forward probabilities for an observation sequence with scaling. 
        Probability of observing the sequence up to time t and being in a specific state at t.

        Input:
            obs (list): A sequence of observations (list of integers).

        Output:
            tuple:
                - alpha (list): Scaled forward probabilities for each time step and state.
                - c (list): Scaling factors applied at each time step.
        """
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
            c[t] = (1 / c[t])
            for i in range(N):
                alpha[t][i] *= c[t]

        return alpha, c

    def backward(self, obs, c):
        """
        Computes the backward probabilities for an observation sequence with scaling.
        Probability of observing the sequence from time  t+1  to the end, starting from a specific state at t.
        
        Input:
            obs (list): A sequence of observations (list of integers).
            c (list): Scaling factors applied at each time step.

        Output:
            beta (list): Scaled backward probabilities for each time step and state.
        """
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
        """
        Computes gamma and di-gamma probabilities for an observation sequence.
        Combination  \alpha_t(i) \cdot \beta_t(i)  gives the probability of being in state i at time t, given the entire observation sequence.

        Input:
            obs (list): A sequence of observations (list of integers).
            alpha (list): Forward probabilities.
            beta (list): Backward probabilities.

        Output:
            tuple:
                - gamma (list): State occupation probabilities for each time step.
                - di_gamma (list): Transition probabilities for each time step.
        """
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
        """
        Updates the model parameters (A, B, pi) based on observation data and gamma values.

        Input:
            obs (list): A sequence of observations (list of integers).
            gamma (list): State occupation probabilities.
            di_gamma (list): Transition probabilities.

        Output:
            None
            - Updates the transition matrix, emission matrix, and initial state probabilities.
        """
        T = len(obs)
        # use epsilon to prevent from zero division error 
        epsilon = 1e-6

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
        """
        Trains the HMM using the Baum-Welch algorithm for a given observation sequence.

        Input:
            obs (list): A sequence of observations (list of integers).

        Output:
            None
            - Updates the model parameters to maximize the likelihood of the observation sequence.
        """
        maxIters = 100
        oldLogProb = -math.inf

        for _ in range(maxIters):

            alpha, c = self.forward(obs)
            beta = self.backward(obs, c)
            gamma, di_gamma = self.compute_gammas(obs, alpha, beta)

            # Re-estimate lambda
            self.update_model(obs, gamma, di_gamma)

            # Compute log-probability
            logProb = 0
            for i in range(len(obs)):
                logProb += math.log(c[i])
            logProb = -logProb

            # Check convergence criterion
            if logProb <= oldLogProb:
                break
            
            oldLogProb = logProb


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    """
    Implements the player controller using Hidden Markov Models (HMMs) to guess the species of fish based on observed movement patterns.

    Attributes:
        hmm_models (list): A list of HMM instances, one for each species, modeling the observations (movements) for each species.
        fish_obs (list): A list of tuples, where each tuple contains a fish ID and its corresponding list of observed movements.
        curr_obs (list): The observation sequence of the most recently guessed fish.
    """
    def init_parameters(self):
        """
        Initializes the parameters needed for the HMM-based fish classification.

        Input:
            None

        Output:
            None
            - Initializes `hmm_models` with one HMM per species (each having 1 state (swimming) and 8 emissions (possible observations)).
            - Initializes `fish_obs` to hold observations for each fish species.
            - Sets `curr_obs` to None.
        """
        self.hmm_models = [HMM(1, 8) for _ in range(N_SPECIES)] # HMM model for each species
        self.fish_obs = [(i, []) for i in range(N_FISH)] # List of fish observations by species
        self.curr_obs = None # Observation sequence of the last guessed fish

    def guess(self, step, observations):
        """
        Guesses the species of a fish based on observed movements using HMMs.

        Input:
            step (int): The current step in the game or process.
            observations (list): A list of observed movements for all fish at the current step.

        Output:
            None or tuple:
                - Returns None if less than 100 steps have occurred.
                - Returns a tuple `(fish_id, fish_type)` after 100 steps:
                    - fish_id (int): The ID of the fish being classified.
                    - fish_type (int): The guessed species of the fish (index of the HMM with the highest probability).
        """
        for i in range(len(self.fish_obs)):
            self.fish_obs[i][1].append(observations[i]) # Store observations for each fish

        if step < 100: # Do not start guessing until step 100
            return None

        # Retrieve the most recent fish ID and its observations
        fish_id, obs = self.fish_obs.pop()
        self.curr_obs = obs

        fish_type = 0
        max_prob = 0

        # Evaluate the observation sequence for each species' HMM
        for i in range(N_SPECIES):
            model = self.hmm_models[i]
            prob = model.evaluate(obs)
            if prob > max_prob: # Update the guess if a higher probability is found
                fish_type = i
                max_prob = prob
        
        return fish_id, fish_type # Return the guessed species with the highest probability 

    def reveal(self, correct, _, true_type):
        """
        Method is called whenever a guess is made.
        It updates the HMM when a guess was made and it was wrong, updating the corresponding species model. 
        :param correct: True if the guess was correct, False otherwise
        :param fish_id: Index of the fish that was guessed
        :param true_type: The correct type of the fish
        """
        # Update the HMM for the true species
        if not correct:
            model = self.hmm_models[true_type]
            model.train(self.curr_obs)
