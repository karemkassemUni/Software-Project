import sys
import numpy as np

np.random.seed(1234)

try:
    import symnmf
except ImportError:
    print("An Error Has Occurred")
    sys.exit(1)


def read_data(filename):
    try:
        points = []
        with open(filename, 'r') as f:
            for line in f:
                points.append([float(x) for x in line.strip().split(',')])
        return points
    except:
        print("An Error Has Occurred")
        sys.exit(1)


def print_matrix(matrix, header=None):
    if header:
        print(header)
    for row in matrix:
        print(','.join(f'{x:.4f}' for x in row))


def calculate_symnmf(points, n, d, k):
    # Calculate similarity matrix
    sim = symnmf.sym(points, n, d)
    if sim is None:
        raise RuntimeError()

    # Calculate degree matrix
    ddg = symnmf.ddg(sim, n)
    if ddg is None:
        raise RuntimeError()

    # Calculate normalized similarity
    w = symnmf.norm(sim, ddg, n)
    if w is None:
        raise RuntimeError()

    # Initialize H
    m = np.mean([np.mean(row) for row in w])
    h_init = [[np.random.uniform(0, 2 * np.sqrt(m / k)) for _ in range(k)] for _ in range(n)]

    # Print H_init
    print_matrix(h_init, "H_init:")

    # Perform factorization
    h_final = symnmf.factorize(w, h_init, n, k)
    if h_final is None:
        raise RuntimeError()

    print_matrix(h_final, "\nH_final:")
    return h_final


def main():
    # Validate command line arguments
    if len(sys.argv) != 4:
        print("An Error Has Occurred, function need 3 args")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]

        # Read input data
        points = read_data(filename)
        if not points:
            raise RuntimeError()

        n = len(points)
        d = len(points[0])

        if not all(len(row) == d for row in points):
            raise RuntimeError()

        # Process based on goal
        if goal == 'symnmf':
            calculate_symnmf(points, n, d, k)
        elif goal == 'sym':
            print(type(points), type(n), type(d))
            result = symnmf.sym(points, n, d)
            print_matrix(result)
        elif goal == 'ddg':
            sim = symnmf.sym(points, n, d)
            result = symnmf.ddg(sim, n)
            print_matrix(result)
        elif goal == 'norm':
            sim = symnmf.sym(points, n, d)
            ddg = symnmf.ddg(sim, n)
            result = symnmf.norm(sim, ddg, n)
            print_matrix(result)
        else:
            raise RuntimeError()

    except:
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
