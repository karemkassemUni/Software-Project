import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import symnmf


def read_data(filename):
    with open(filename, 'r') as f:
        points = np.array([list(map(float, line.strip().split(','))) for line in f])
    return points


def get_clusters_from_h(h):
    """Convert H matrix to cluster assignments based on maximum values"""
    return np.argmax(h, axis=1)


def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        return

    k = int(sys.argv[1])
    filename = sys.argv[2]

    try:
        # Read data
        points = read_data(filename)
        points_lst = points.tolist()
        n = len(points)
        d = len(points[0])
        # SymNMF clustering
        # Calculate similarity matrix

        sim = symnmf.sym(points_lst, n, d)

        # Calculate degree matrix
        ddg = symnmf.ddg(sim, n)

        # Calculate normalized similarity
        w = symnmf.norm(sim, ddg, n)

        # Initialize H
        np.random.seed(1234)
        m = np.mean(w)
        h_init = [[np.random.uniform(0, 2 * np.sqrt(m / k)) for _ in range(k)] for _ in range(n)]

        # Get final H matrix

        h = symnmf.factorize(w, h_init, n, k)

        # Get cluster assignments
        nmf_clusters = get_clusters_from_h(h)

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=1234, n_init=10)

        kmeans_clusters = kmeans.fit_predict(points)

        # Calculate silhouette scores
        nmf_score = silhouette_score(points, nmf_clusters)
        kmeans_score = silhouette_score(points, kmeans_clusters)

        # Print results
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception as e:
        print("An Error Has Occurred")
        return


if __name__ == "__main__":
    main()
