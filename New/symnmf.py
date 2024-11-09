import sys
import logging
import numpy as np
np.random.seed(1234)

try:
    import symnmf
except ImportError:
    print("Error: Could not import symnmf module.", file=sys.stderr)
    sys.exit(1)

def read_data(filename):
    try:
        points = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    row = [float(x) for x in line.split(',')]
                    points.append(row)
        return points
    except Exception as e:
        print(f"Error reading file: {str(e)}", file=sys.stderr)
        raise

def print_matrix(matrix, header=None):
    if header:
        print(header)
    for row in matrix:
        print(','.join(f'{x:.4f}' for x in row))

def validate_input(points):
    if not points or not points[0]:
        return False
    length = len(points[0])
    return all(len(row) == length for row in points)


def print_intermediate_h(h):
    print("Intermediate H:")    
    for row in h:
        print(','.join(f'{x:.4f}' for x in row))

def main():
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)
    
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]
        
        points = read_data(filename)
        if not validate_input(points):
            print("An Error Has Occurred")
            sys.exit(1)
            
        n = len(points)
        d = len(points[0])
        
        if goal == 'symnmf':
            # Calculate similarity matrix
            sim = symnmf.sym(points, n, d)
            if sim is None:
                print("An Error Has Occurred")
                sys.exit(1)
            
            # Calculate degree matrix
            ddg = symnmf.ddg(sim, n)
            if ddg is None:
                print("An Error Has Occurred")
                sys.exit(1)
            
            # Calculate normalized similarity
            w = symnmf.norm(sim, ddg, n)
            if w is None:
                print("An Error Has Occurred")
                sys.exit(1)
            
            # Initialize H
            m = np.mean([np.mean(row) for row in w])
            h_init = [[np.random.uniform(0, 2 * np.sqrt(m/k)) for _ in range(k)] for _ in range(n)]
            
            # Convert h_init to numpy array for better manipulation
            h_init = np.array(h_init)
            
            # Print H_init
            print_matrix(h_init.tolist(), "H_init")
            
            # Perform factorization
            print("W matrix:")
            print_matrix(w)
            print("\nH_init matrix:")
            print_matrix(h_init.tolist())


            h_final = symnmf.factorize(w, h_init.tolist(), n, k)
            if h_final is None:
                print("An Error Has Occurred")
                sys.exit(1)
            
            print_intermediate_h(h_final)  # Add this line to print intermediate H matrix
            
            # Verify h_final is different from h_init
            if np.array_equal(np.array(h_final), h_init):
                print("Warning: H matrix did not change during optimization")
            
            # Print H_final
            print_matrix(h_final, "H_final")
            
        else:
            # Handle other cases (sym, ddg, norm)...
            result = None
            if goal == 'sym':
                result = symnmf.sym(points, n, d)
            elif goal == 'ddg':
                sim = symnmf.sym(points, n, d)
                result = symnmf.ddg(sim, n)
            elif goal == 'norm':
                sim = symnmf.sym(points, n, d)
                ddg = symnmf.ddg(sim, n)
                result = symnmf.norm(sim, ddg, n)
            
            if result is None:
                print("An Error Has Occurred")
                sys.exit(1)
            print_matrix(result)
            
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()