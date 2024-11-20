import sys
import numpy as np
# Calculate emission probability distribution 


def execute_hmm():
    # Read and parse input data
    transition_data = sys.stdin.readline().split()
    emission_data = sys.stdin.readline().split()
    initial_data = sys.stdin.readline().split()

    # Convert the input data into matrices
    # A - states
    transition_matrix = convert_to_matrix(transition_data[2:], int(transition_data[0]), int(transition_data[1]))
    # B - observations
    emission_matrix = convert_to_matrix(emission_data[2:], int(emission_data[0]), int(emission_data[1]))
    # pi 
    initial_distribution = convert_to_matrix(initial_data[2:], int(initial_data[0]), int(initial_data[1]))

    # Perform matrix multiplications
    # pi' = pi * A
    pi_times_transition = manual_dot(initial_distribution, transition_matrix)
    # R = pi' * B
    final_result = manual_dot(pi_times_transition, emission_matrix)

    # Format the result
    output = f"{len(final_result)} {len(final_result[0])} "
    output += ' '.join([str(round(value, 6)) for row in final_result for value in row])

    return output

def convert_to_matrix(data, num_rows, num_cols):

    data = list(map(float, data))
    return [data[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

def manual_dot(matrix_a, matrix_b):
    """
    Perform matrix multiplication between two matrices (lists of lists).
    
    Parameters:
    - matrix_a: List of lists, representing matrix A (m x n)
    - matrix_b: List of lists, representing matrix B (n x p)
    
    Returns:
    - result: List of lists, representing the result of the dot product (m x p)
    """
    # Get dimensions of the matrices
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])
    
    # Ensure matrices are compatible for multiplication
    if cols_a != rows_b:
        raise ValueError("Matrix A's number of columns must match Matrix B's number of rows")
    
    # Initialize result matrix with zeros (m x p)
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # Perform the dot product
    for i in range(rows_a):  # Iterate through rows of A
        for j in range(cols_b):  # Iterate through columns of B
            for k in range(cols_a):  # Iterate through elements to multiply
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result

if __name__ == "__main__":
    print(execute_hmm())
    
    