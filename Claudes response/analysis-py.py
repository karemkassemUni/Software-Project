import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import symnmf

def read_data(filename):
    """Read data points from file."""
    try:
        with open(filename, 'r') as f:
            return np.array([list(map(float, line.strip().split(','))) for line in f])
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

def get_clusters_from_h(h):
    """Convert H matrix to cluster assignments."""
    return np.argmax(h, axis=1)

def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        filename = sys.argv[2]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    # Read data
    X = read_data(filename)
    
    if k >= len(X):
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        # Perform SymNMF clustering
        W = np.array(symnmf.norm(X.tolist()))
        np.random.seed(1234)
        m = np.mean(W)
        H_init = np.random.uniform(0, 2 * np.sqrt(m/k), (len(X), k))
        H = np.array(symnmf.symnmf(H_init.tolist(), W.tolist()))
        symnmf_labels = get_clusters_from_h(H)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=1234)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette scores
        symnmf_score = silhouette_score(X, symnmf_labels)
        kmeans_score = silhouette_score(X, kmeans_labels)
        
        # Print results
        print(f"nmf: {symnmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()
