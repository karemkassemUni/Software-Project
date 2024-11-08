import sys
import logging
import numpy as np

try:
    import symnmf
except ImportError:
    print("Error: Could not import symnmf module. Did you run 'python3 setup.py build_ext --inplace'?", file=sys.stderr)
    sys.exit(1)

def read_data(filename):
    try:
        points = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    row = [float(x) for x in line.split(',')]
                    points.append(row)
        return points
    except Exception as e:
        print(f"Error reading file: {str(e)}", file=sys.stderr)
        raise

def print_matrix(matrix):
    for row in matrix:
        print(','.join(f'{x:.4f}' for x in row))

def validate_input(points):
    if not points or not points[0]:
        return False
    length = len(points[0])
    return all(len(row) == length for row in points)

def main():
    # Input validation
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)
    
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]
        
        # Read and validate data points
        points = read_data(filename)
        if not validate_input(points):
            print("An Error Has Occurred")
            sys.exit(1)
            
        n = len(points)
        d = len(points[0])
        
        if goal == 'sym':
            result = symnmf.sym(points, n, d)
        elif goal == 'ddg':
            sim = symnmf.sym(points, n, d)
            result = symnmf.ddg(sim, n)
        elif goal == 'norm':
            sim = symnmf.sym(points, n, d)
            ddg = symnmf.ddg(sim, n)
            result = symnmf.norm(sim, ddg, n)
        elif goal == 'symnmf':
            sim = symnmf.sym(points, n, d)
            ddg = symnmf.ddg(sim, n)
            w = symnmf.norm(sim, ddg, n)
            
            # Initialize H
            np.random.seed(1234)
            m = sum(sum(row) for row in w) / (n * n)
            h_init = [[np.random.uniform(0, 2 * (m/k)**0.5) for _ in range(k)] for _ in range(n)]
            
            result = symnmf.factorize(w, h_init, n, k)
        else:
            print("An Error Has Occurred")
            sys.exit(1)
            
        print_matrix(result)
        
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()