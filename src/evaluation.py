import json
import pandas as pd
import paths

ground_truth = pd.read_csv(paths.GROUND_TRUTH_DIR + "/monitor_entity_resolution_gt(in).csv")

def generate_entity2clusters():
    # Group by entity_id and get a list of spec_id
    group_by_entity_id = ground_truth.groupby('entity_id')['spec_id'].apply(list)

    clusters: dict[str, list[str]] = json.load(open(paths.RESULTS_DIR + "/clustering/dbscan/dbscan_clusters.json"))

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

    json.dump(entity2clusters, open(paths.RESULTS_DIR + "/evaluation/entity2clusters.json", 'w'), indent=4)
        