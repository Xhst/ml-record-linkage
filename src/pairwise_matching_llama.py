import json
import paths
from itertools import combinations
from llama_cpp import Llama

def load_clusters(file_path):
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    return clusters

def load_item2pagetitle(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        item2pagetitle: dict[str,str] = json.load(f)
    return item2pagetitle


def query_llama3(llm, title1, title2, begin: bool):
    '''
    Queries the LLAMA model to determine if two webpage titles represent the same object.
    
    Args:
        llm (LLMModel): The LLAMA model used for inference.
        title1 (str): The first webpage title.
        title2 (str): The second webpage title.
        begin (bool): Flag indicating whether to include the "<|begin_of_text|>" token in the prompt.
    
    Returns:
        str: The model's response indicating whether the webpage titles represent the same object ("yes" or "no").
    '''
    # Construct the prompt
    # Generate response from the LLAMA model
    output = llm(prompt, max_tokens=30, temperature=0.15, echo=False)
    prompt = ""
    if begin:
      prompt += "<|begin_of_text|>"
    prompt += f'''<|start_header_id|>system<|end_header_id|>
    You are an helful assistant that can tell if two monitors are the same object just by analyzing their product webpage title.
    To help yourself search for model names or alphanumerical strings and try to ignore the webpage name in the webpage titles.
    If the page titles represent the same entity your answer MUST BE "yes".
    If the page titles do not represent the same entity your answer MUST BE "no".

    Example 1:
    first page title: "Hp Hewlett Packard HP Z22i D7Q14AT ABB Planet Computer.it"
    second page title: "HP Z22i - MrHighTech Shop"
    "yes" 
    Example 2:
    first page title: "Hp Hewlett Packard HP Z22i D7Q14AT ABB Planet Computer.it"
    second page title: "C4D33AA#ABA - Hp Pavilion 20xi Ips Led Backlit Monitor - PC-Canada.com"
    "no"
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Now tell me if these two webpage titles represent the same object:
    first webpage title: "{title1}"
    second webpage title: "{title2}"
    Answers MUST BE "yes" or "no"
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

    output = llm(prompt, max_tokens=30, temperature=0.15, echo=False )
    response = output['choices'][0]['text']
    print(response)
    return response


def pairwise_matching(clusters: dict[str, list[str]], item2pagetitle: dict[str,str], num_clusters_to_analyze: int):
    '''
    Perform pairwise matching between items in the clusters.

    Args:
        clusters (dict[str, list[str]]): A dictionary containing the clusters of items.
        item2pagetitle (dict[str,str]): A dictionary mapping item IDs to webpage titles.
        num_clusters_to_analyze (int): The number of clusters to analyze.
    
    Returns:
        dict[str, list[tuple[str,str,str]]]: A dictionary containing the matches for each cluster.
    '''
    # for each cluster we have a list of tuples (page_title1, page_title2, match_result)
    matches: dict[str, list[tuple[str,str,str]]] = {}
    
    llm = Llama(model_path= paths.MODELS_DIR + "/Llama3-70B/L3-70B-Euryale-v2.1-IQ3_XXS.gguf")
    
    begin = True
    
    for cluster_id, items in clusters.items():
        if num_clusters_to_analyze == 0:
            break
        matches[cluster_id] = []
        # Creazione delle coppie
        item_pairs = combinations(items, 2)
        for item1, item2 in item_pairs:
            title1 = item2pagetitle[item1]
            title2 = item2pagetitle[item2]
            match_status = query_llama3(llm, title1, title2, begin)
            result = (item1, item2, match_status)
            print(result)
            matches[cluster_id].append(result)
            begin = False
        num_clusters_to_analyze -= 1
    
    return matches


def save_matches(matches, output_file):
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=4)


def calculate_combinations_for_cluster(cluster):
    return len(list(combinations(cluster, 2)))


def calculate_combinations_for_clusters(clusters):
    return sum(calculate_combinations_for_cluster(cluster) for cluster in clusters.values())


def main():
    clusters_file = paths.RESULTS_DIR + "/clustering/hdbscan/hdbscan_clusters.json"
    clusters: dict[str, list[str]] = load_clusters(clusters_file)
  
    # Remove the cluster with ID "-1" because it's the cluster of outliers
    del clusters["-1"]
    
    print(f"Calculating the number of combinations to be performed (on {str(len(clusters))} clusters): {calculate_combinations_for_clusters(clusters)}")
    
    item2pagetitle_file = paths.RESULTS_DIR + "/preprocessing/item2pagetitle.json"
    item2pagetitle: dict[str,str] = load_item2pagetitle(item2pagetitle_file)
    
    num_clusters_to_analyze = 1
    matches = pairwise_matching(clusters, item2pagetitle, num_clusters_to_analyze)
    
    output_file = paths.RESULTS_DIR + "/pairwise_matching/matches.json"
    save_matches(matches, output_file)

if __name__ == "__main__":
    main()
