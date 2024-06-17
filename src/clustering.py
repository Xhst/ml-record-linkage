import json
import argparse
import time
import torch
from sklearn.cluster import DBSCAN, KMeans
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering

def load_embeddings(file_dir, file_name):
    """
    Carica gli embedding da un file JSON.
    """
    with open(file_dir + "/" + file_name, 'r') as f:
        embeddings = json.load(f)
    return embeddings

def cluster_embeddings(embeddings, algorithm, **kwargs):
    """
    Esegue il clustering sugli embedding utilizzando l'algoritmo specificato.
    """
    embeddings_array = torch.tensor([value for value in embeddings.values()])
    if algorithm == 'hdbscan':
        clusterer = HDBSCAN(metric='euclidean', **kwargs)
        cluster_labels = clusterer.fit_predict(embeddings_array.numpy())
    elif algorithm == 'kmeans':
        clusterer = KMeans(**kwargs)
        cluster_labels = clusterer.fit_predict(embeddings_array.numpy())
    elif algorithm == 'dbscan':
        clusterer = DBSCAN(metric='euclidean', **kwargs)
        cluster_labels = clusterer.fit_predict(embeddings_array.numpy())
    elif algorithm == 'agglomerative':
        clusterer = AgglomerativeClustering(metric='euclidean', **kwargs)
        cluster_labels = clusterer.fit_predict(embeddings_array.numpy())
    else:
        raise ValueError(f"Algoritmo di clustering non valido: {algorithm}")
    return cluster_labels

def save_clusters(item2embedding, cluster_labels, output_dir, output_file):
    """
    Salva i risultati del clustering in un file JSON.
    """
    print("Saving clusters to JSON...")
    clusters = {}
    for i, label in enumerate(cluster_labels):
        filename = list(item2embedding.keys())[i]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)

    # Convert cluster labels to int (if necessary) for JSON serialization
    clusters = {int(k): v for k, v in clusters.items()}

    with open(output_dir + "/" + output_file, 'w') as f:
        json.dump(clusters, f, indent=4)

    print(f"Clusters saved successfully to " + output_dir + "/" + output_file)

def main():
    parser = argparse.ArgumentParser(description='Clustering di embedding')
    parser.add_argument('--embeddings_dir', default='results/embeddings/', help='Directory degli embedding')
    parser.add_argument('--embeddings_file', default='embeddings_distilbert_base_uncased.json', help='File degli embedding')
    parser.add_argument('--algorithm', default='hdbscan', choices=['hdbscan', 'kmeans', 'dbscan', 'agglomerative'], help='Algoritmo di clustering')
    parser.add_argument('--output_dir', default='results/clustering/', help='Directory di output')
    parser.add_argument('--output_file', default='clusters.json', help='File di output')
    args = parser.parse_args()

    item2embedding = load_embeddings(args.embeddings_dir, args.embeddings_file)
    print("Embeddings loaded successfully")

    start = time.time()

    if args.algorithm == 'hdbscan':
        cluster_labels = cluster_embeddings(item2embedding, args.algorithm, min_cluster_size=5, min_samples=1)
    elif args.algorithm == 'kmeans':
        cluster_labels = cluster_embeddings(item2embedding, args.algorithm, n_clusters=5)
    elif args.algorithm == 'dbscan':
        cluster_labels = cluster_embeddings(item2embedding, args.algorithm, eps=0.5, min_samples=10)
    elif args.algorithm == 'agglomerative':
        cluster_labels = cluster_embeddings(item2embedding, args.algorithm, n_clusters=5)

    print("Clustering completed")

    save_clusters(item2embedding, cluster_labels, args.output_dir, args.output_file)

    end = time.time()
    print(f"CLUSTERING --- Processing time: {end - start:.2f} seconds")

if __name__ == "__main__":
    print("Elaboration starting...")
    main()