import json
import paths
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.input_layer = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def forward_once(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
    

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive


class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]
    

class SiameseTraining:
    def __init__(self, model, dataloader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        

    def train(self, epochs):
        for epoch in range(epochs):
            for data in self.dataloader:
                (input1, input2), label = data
                
                self.optimizer.zero_grad()
                output1, output2 = self.model(input1, input2)
                loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    

def generate_pairs(entity2clusters, embeddings):
    pairs = []
    labels = []
    
    for entity_id, clusters in entity2clusters.items():
        for _, items in clusters.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)): 
                    pair1 = torch.tensor(embeddings[items[i]]).float()
                    pair2 = torch.tensor(embeddings[items[j]]).float()
                    pairs.append((pair1, pair2))
                    labels.append(1)
                    
            for other_entity_id, other_clusters in entity2clusters.items():
                if other_entity_id == entity_id:
                    continue
                
                for _, other_items in other_clusters.items():
                    for item in items:
                        for other_item in other_items:
                            pair1 = torch.tensor(embeddings[item]).float()
                            pair2 = torch.tensor(embeddings[other_item]).float()
                            pairs.append((pair1, pair2))
                            labels.append(0)
    
    return pairs, labels




