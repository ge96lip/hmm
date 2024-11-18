import sys
import numpy as np

def execute_hmm():
    # Read and parse input data
    transition_data = sys.stdin.readline().split()
    emission_data = sys.stdin.readline().split()
    initial_data = sys.stdin.readline().split()

    # Convert the input data into matrices
    transition_matrix = convert_to_matrix(transition_data[2:], int(transition_data[0]), int(transition_data[1]))
    emission_matrix = convert_to_matrix(emission_data[2:], int(emission_data[0]), int(emission_data[1]))
    initial_distribution = convert_to_matrix(initial_data[2:], int(initial_data[0]), int(initial_data[1]))

    # Perform matrix multiplications
    pi_times_transition = np.dot(initial_distribution, transition_matrix).tolist() 
    final_result = np.dot(pi_times_transition, emission_matrix).tolist()  

    # Format the result
    output = f"{len(final_result)} {len(final_result[0])} "
    output += ' '.join([str(round(value, 6)) for row in final_result for value in row])

    return output

def convert_to_matrix(data, num_rows, num_cols):

    data = list(map(float, data))
    return [data[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

if __name__ == "__main__":
    print(execute_hmm())