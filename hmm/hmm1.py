import sys
import numpy as np

def forward(prev_alpha, emission_sequence, A, B): 
    if not emission_sequence:
        final_sum = sum(prev_alpha)
        print(round(final_sum, 6))
        return final_sum
    
    next_alpha = np.dot(prev_alpha, A)  # Matrix-vector multiplication: prev_alpha * A
    next_alpha *= B[:, emission_sequence[0]]  # Element-wise multiplication with the emission probabilities

    # Recursively call the function with the updated alpha vector and the remaining emission sequence
    return forward(next_alpha, emission_sequence[1:], A, B)

def execute_hmm():
    # Read and parse input data
    transition_data = sys.stdin.readline().split()
    emission_data = sys.stdin.readline().split()
    initial_data = sys.stdin.readline().split()
    seq_emission = [int(x) for x in sys.stdin.readline().split()[1:]]
    
    # Convert the input data into matrices
    transition_matrix = convert_to_matrix(transition_data[2:], int(transition_data[0]), int(transition_data[1]))
    emission_matrix = convert_to_matrix(emission_data[2:], int(emission_data[0]), int(emission_data[1]))
    initial_distribution = convert_to_matrix(initial_data[2:], int(initial_data[0]), int(initial_data[1]))
    
    emission_matrix = np.array(emission_matrix)
    #print(emission_data)
    initial_alpha = (initial_distribution[0] * emission_matrix[:, seq_emission[0]]).tolist()
    
    forward(initial_alpha, seq_emission[1:], transition_matrix, emission_matrix)
    
def convert_to_matrix(data, num_rows, num_cols):

    data = list(map(float, data))
    return [data[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]
    
if __name__ == "__main__":
    execute_hmm()