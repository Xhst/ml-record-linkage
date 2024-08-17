import json
import pandas as pd
import paths
import torch
import torch.nn as nn

ground_truth = pd.read_csv(paths.GROUND_TRUTH_DIR + "/monitor_entity_resolution_gt(in).csv")

def generate_entity2clusters(algorithm: str = "hdbscan"):
    # Group by entity_id and get a list of spec_id
    group_by_entity_id = ground_truth.groupby('entity_id')['spec_id'].apply(list)

    clusters: dict[str, list[str]] = json.load(open(paths.RESULTS_DIR + "/clustering/" + algorithm + "/" + algorithm + "_clusters.json"))

    entity2clusters: dict[str, dict[str, list[str]]] = {}

    # For each cluster, find the items that belong to the same entity
    for cluster_id, cluster in clusters.items():
        for item in cluster:
            try:
                # Find the entity that contains the item, if it exists
                entity_id = group_by_entity_id[group_by_entity_id.apply(lambda x: item in x)].index[0]
            except:
                continue
            
            if entity_id not in entity2clusters:
                # create a dict: cluster_id -> items for every entity
                entity2clusters[entity_id] = {}
                
            if cluster_id not in entity2clusters[entity_id]:
                # create a list of items for each cluster (for every entity)
                entity2clusters[entity_id][cluster_id] = []
                
            entity2clusters[entity_id][cluster_id].append(item)

    # order by enitity_id
    entity2clusters = {k: v for k, v in sorted(entity2clusters.items(), key=lambda item: item[0])}

    json.dump(entity2clusters, open(paths.RESULTS_DIR + "/evaluation/" + algorithm + "_entity2clusters.json", 'w'), indent=4)



### SIAMESE NETWORK ###

def predict_match_using_distance(output1: torch.Tensor, output2: torch.Tensor, threshold: float = 0.5):
    '''
    Predict match or not for a batch of inputs based on the Euclidean distance between outputs.
    
    Args:
        output1 (torch.Tensor): output tensor of the first network (batch size, embedding size)
        output2 (torch.Tensor): output tensor of the second network (batch size, embedding size)
        threshold (float): distance threshold to classify as match
    
    Returns:
        matches (torch.Tensor): tensor of binary responses (1 for match, 0 for not match)
        distances (torch.Tensor): tensor of Euclidean distances
    '''
    # Compute Euclidean distances for the entire batch
    euclidean_distances = nn.functional.pairwise_distance(output1, output2)
    
    # Apply the threshold to determine if each pair is a match or not
    matches = (euclidean_distances <= threshold).int()  # Convert to 0 or 1 for match
    
    return matches, euclidean_distances


def evaluate_siamese_model(model, dataloader, threshold=0.5, enable_prints=False):
    '''
    Evaluate the Siamese model on a validation set and calculate precision, recall, and F1-score.
    
    Args:
        model (nn.Module): trained Siamese model
        dataloader (DataLoader): validation data loader
        threshold (float): distance threshold to classify as match
    
    Returns:
        precision (float): precision score
        recall (float): recall score
        f1_score (float): F1 score
    '''
    TP, FP, FN = 0, 0, 0  # Initialize true positives, false positives, false negatives
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data in dataloader:
            (input1, input2), labels = data  # Get the pairs and the ground truth labels
            output1, output2 = model(input1, input2)  # Get the model outputs
            matches, distances = predict_match_using_distance(output1, output2, threshold)  # Predict matches
            
            # Convert labels and matches to CPU tensors if needed
            labels = labels.cpu()
            matches = matches.cpu()

            # Iterate over the batch
            for i in range(len(matches)):
                # True Positive: Model predicts match and the label is 1
                if matches[i] == 1 and labels[i] == 1:
                    if enable_prints:
                        print(f"true positive: {matches[i], labels[i], distances[i]}")
                    TP += 1
                # False Positive: Model predicts match but the label is 0
                elif matches[i] == 1 and labels[i] == 0:
                    if enable_prints and i % 10000 == 0:
                        print(f"false positive: {matches[i], labels[i], distances[i]}")
                    FP += 1
                # False Negative: Model predicts no match but the label is 1
                elif matches[i] == 0 and labels[i] == 1:
                    FN += 1
    
    # Calculate precision, recall, and F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score
