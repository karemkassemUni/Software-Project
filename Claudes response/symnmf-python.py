import numpy as np
import sys
import symnmf  # This will be our C extension module

def read_data(filename):
    """Read data points from file."""
    try:
        with open(filename, 'r') as f:
            return np.array([list(map(float, line.strip().split(','))) for line in f])
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

def initialize_h(w, k):
    """Initialize H matrix as described in section 1.4.1."""
    np.random.seed(1234)
    n = len(w)
    m = np.mean(w)
    return np.random.uniform(0, 2 * np.sqrt(m/k), (n, k))

def format_matrix(matrix):
    """Format matrix output according to specifications."""
    return '\n'.join([','.join([f'{x:.4f}' for x in row]) for row in matrix])

def main():
    # Validate command line arguments
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    # Read data points
    X = read_data(filename)
    
    # Validate k
    if k >= len(X):
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        if goal == "symnmf":
            # Get normalized similarity matrix from C extension
            W = np.array(symnmf.norm(X.tolist()))
            
            # Initialize H
            H_init = initialize_h(W, k)
            
            # Call C extension for symNMF
            H = np.array(symnmf.symnmf(H_init.tolist(), W.tolist()))
            
            print(format_matrix(H))

        elif goal == "sym":
            # Calculate similarity matrix using C extension
            A = np.array(symnmf.sym(X.tolist()))
            print(format_matrix(A))

        elif goal == "ddg":
            # Calculate diagonal degree matrix using C extension
            D = np.array(symnmf.ddg(X.tolist()))
            print(format_matrix(D))

        elif goal == "norm":
            # Calculate normalized similarity matrix using C extension
            W = np.array(symnmf.norm(X.tolist()))
            print(format_matrix(W))

        else:
            print("An Error Has Occurred")
            sys.exit(1)

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()
