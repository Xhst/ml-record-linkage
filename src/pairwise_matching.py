import json
import requests
import argparse

def load_clusters(file_path):
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    return clusters

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        embeddings = json.load(f)
    return embeddings

# FIXME
def query_llm(text1, text2, model="mistral"):
    api_url = "https://api-inference.huggingface.co/models/" + model
    headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_API_KEY"}
    payload = {
        "inputs": {
            "source_sentence": text1,
            "sentences": [text2]
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    result = response.json()
    similarity_score = result[0]['score']
    return similarity_score

def pairwise_matching(clusters, embeddings, threshold=0.8):
    matches = []
    for cluster in clusters.values():
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                file1, file2 = cluster[i], cluster[j]
                text1, text2 = embeddings[file1], embeddings[file2]
                similarity_score = query_llm(text1, text2)
                if similarity_score > threshold:
                    matches.append((file1, file2, similarity_score))
    return matches

def save_matches(matches, output_file):
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Perform pairwise matching within clusters using an LLM.")
    parser.add_argument("--clusters", type=str, required=True, help="Path to the clusters file")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the matching output")
    args = parser.parse_args()

    clusters = load_clusters(args.clusters)
    embeddings = load_embeddings(args.embeddings)
    matches = pairwise_matching(clusters, embeddings)
    save_matches(matches, args.output)

if __name__ == "__main__":
    main()
