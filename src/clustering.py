import json
import hdbscan
import numpy as np
import argparse
import time
import torch
from tqdm import tqdm

def load_embeddings(file_path):
    """
    Carica gli embedding da un file JSON.
    """
    with open(file_path, 'r') as f:
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

def save_clusters(embeddings, cluster_labels, output_file):
    """
    Salva i risultati del clustering in un file JSON.
    """
    print("Saving clusters to JSON...")
    clusters = {}
    for i, label in tqdm(enumerate(cluster_labels), total=len(cluster_labels), desc="Saving clusters"):
        filename = list(embeddings.keys())[i]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)
    
    # Convert cluster labels to int (if necessary) for JSON serialization
    clusters = {int(k): v for k, v in clusters.items()}
    
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=4)
    
    print(f"Clusters saved successfully to {output_file}")

def assign_device(device_name):
    if device_name == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif device_name == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS is available. Using Apple Silicon GPU.")
        else:
            print("MPS is not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    # Display available devices
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
   
    return device

def main():
    """
    Punto di ingresso principale per il clustering degli embedding e il salvataggio dei risultati.
    """
    parser = argparse.ArgumentParser(description="Cluster embeddings and save the resulting clusters.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the clustered output")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Computing device to use (cpu, cuda, or mps)")
    args = parser.parse_args()
    device = assign_device(args.device)
    
    embeddings = load_embeddings(args.embeddings)
    print("Embeddings loaded successfully")
    start = time.time()

    with tqdm(total=len(embeddings), desc="Clustering") as pbar:
        cluster_labels = cluster_embeddings(embeddings)
        pbar.update(len(embeddings))

    print("Clustering completed")
    save_clusters(embeddings, cluster_labels, args.output)
    print("Clusters saved successfully")
    end = time.time()
    print(f"Processing time (using {device}): {end - start:.2f} seconds")

if __name__ == "__main__":
    print("Elaboration starting...")
    main()
