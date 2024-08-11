import json
import paths
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        '''
        Siamese network
        
        Args:
            embedding_dim (int): embedding (input layer) dimension
            hidden_dim (int): hidden layer dimension
        '''
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.input_layer = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass in a single network

        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            x (torch.Tensor): output tensor
        '''
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
    

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the siamese network
        
        Args:
            input1 (torch.Tensor): input tensor for the first network
            input2 (torch.Tensor): input tensor for the second network
            
        Returns:
            output1 (torch.Tensor): output tensor of the first network
            output2 (torch.Tensor): output tensor of the second network
        '''
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        '''
        Contrastive loss function
        
        Args:
            margin (float): margin value
        '''
        super(ContrastiveLoss, self).__init__()

        self.margin = margin


    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: float) -> torch.Tensor:
        '''
        Forward pass of the contrastive loss function
        
        Args:
            output1 (torch.Tensor): output tensor of the first network
            output2 (torch.Tensor): output tensor of the second network
            label (float): label indicating whether the pair is similar or not
            
        Returns:
            loss_contrastive (torch.Tensor): contrastive loss
        '''
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive


class SiameseDataset(Dataset):
    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], labels: list[int]):
        self.pairs = pairs
        self.labels = labels


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx], self.labels[idx]
    

class SiameseTraining:
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        

    def train(self, epochs: int):
        '''
        Train the siamese network

        Args:
            epochs (int): number of epochs
        '''
        print("Starting training")

        for epoch in range(epochs):
            for data in self.dataloader:
                (input1, input2), label = data
                
                self.optimizer.zero_grad()
                output1, output2 = self.model(input1, input2)
                loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    

def generate_pairs(entity2clusters: dict, embeddings: dict) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[int]]:
    '''
    Generate pairs of entities and labels for the siamese network
    
    Args:
        entity2clusters (dict): dictionary containing clusters of entities
        embeddings (dict): dictionary containing embeddings of entities
        
    Returns:
        pairs (list[tuple[torch.Tensor, torch.Tensor]]): list of pairs of entities
        labels (list[int]): list of labels
    '''
    pairs = []
    labels = []
    
    for entity_id, clusters in entity2clusters.items():
        
        # Generate positive pairs
        for _, items in clusters.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)): 
                    pair1 = torch.tensor(embeddings[items[i]]).float()
                    pair2 = torch.tensor(embeddings[items[j]]).float()
                    pairs.append((pair1, pair2))
                    labels.append(1)

        # Generate negative pairs     
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

    print(f"Generated {len(pairs)} pairs")
    
    return pairs, labels


if __name__ == "__main__":
    entity2clusters = json.load(open(paths.RESULTS_DIR + "/evaluation/entity2clusters.json"))
    embeddings = json.load(open(paths.RESULTS_DIR + "/embeddings/embeddings_distilbert_base_uncased.json"))

    pairs, labels = generate_pairs(entity2clusters, embeddings)

    dataset = SiameseDataset(pairs, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SiameseNetwork(768, 256)

    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    training = SiameseTraining(model, dataloader, criterion, optimizer)
    training.train(10)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])



