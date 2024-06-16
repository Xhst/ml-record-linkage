import json
import hdbscan
import numpy as np
import argparse
import time
import torch
from tqdm import tqdm


def load_embeddings(file_dir, file_name):
    """
    Carica gli embedding da un file JSON.
    """
    with open(file_dir + "/" + file_name, 'r') as f:
        embeddings = json.load(f)
    return embeddings


def cluster_embeddings(embeddings, min_cluster_size=5, min_samples=1):
    """
    Esegue il clustering sugli embedding utilizzando HDBSCAN.
    """
    embeddings_array = np.array(list(embeddings.values()))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings_array)
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
    embeddings_dir = "results/embeddings/"
    embeddings_file = "embeddings_distilbert_base_uncased.json"
    
    item2embedding = load_embeddings(embeddings_dir, embeddings_file)
    print("Embeddings loaded successfully")
    
    start = time.time()
    
    cluster_labels = cluster_embeddings(item2embedding)

    print("Clustering completed")
    
    output_dir = "results/clustering/"
    save_clusters(item2embedding, cluster_labels, output_dir, "hdbscan_clusters.json")
   
    end = time.time()
    print(f"CLUSTERING --- Processing time: {end - start:.2f} seconds")


if __name__ == "__main__":
    print("Elaboration starting...")
    main()
