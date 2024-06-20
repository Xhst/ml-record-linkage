import json
import requests
import paths

def load_clusters(file_path):
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    return clusters

def load_item2pagetitle(file_path):
    with open(file_path, 'r') as f:
        item2pagetitle = json.load(f)
    return item2pagetitle


def query_llm(text1, text2, model="mistral"):
    similarity_score = 0.0
    return similarity_score

def pairwise_matching(clusters, item2pagetitle):
    matches = {}
    return matches

def save_matches(matches, output_file):
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=4)

def main():
    clusters_file = paths.RESULTS_DIR + "/clustering/hdbscan/hdbscan_clusters.json"
    clusters: dict[str, list[str]] = load_clusters(clusters_file)
    
    item2pagetitle_file = paths.RESULTS_DIR + "/preprocessing/item2pagetitle.json"
    item2pagetitle: dict[str,str] = load_item2pagetitle(item2pagetitle_file)
    
    matches = pairwise_matching(clusters, item2pagetitle)
    
    output_file = paths.RESULTS_DIR + "/pairwise_matching/matches.json"
    save_matches(matches, output_file)

if __name__ == "__main__":
    main()
