import argparse
import json
import sys
import paths
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from embedder import assign_device


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
    def __init__(self, pairs: list[tuple[str, str]], labels: list[int], embeddings: dict):
        self.pairs = pairs
        self.labels = labels
        self.embeddings = {key: torch.tensor(value, dtype=torch.float) for key, value in embeddings.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        pair = (self.embeddings[self.pairs[idx][0]], self.embeddings[self.pairs[idx][1]])
        return pair, self.labels[idx]
    

class SiameseTraining:
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        

    def train(self, epochs: int, enable_prints: bool = True, print_every: int = 10):
        print("Starting training")
        for epoch in range(epochs):
            for i, data in enumerate(self.dataloader):
                (input1, input2), labels = data
                
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                
                output1, output2 = self.model(input1, input2)
                loss = self.criterion(output1, output2, labels)
                
                loss.backward()
                self.optimizer.step()
                
                if enable_prints and i % print_every == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Item [{i}/{len(self.dataloader)}], Loss: {loss.item():.6f}')
            
            # Save the model at the end of each epoch
            torch.save(self.model.state_dict(), paths.MODELS_DIR + f"/siamese_net/siamese_model_epoch_{epoch}.pth")
            torch.save(self.optimizer.state_dict(), paths.MODELS_DIR + f"/siamese_net/siamese_optimizer_epoch_{epoch}.pth")
            print(f"\033[32mSaved model and optimizer state for epoch {epoch}\033[0m")
    

def generate_pairs(entity2clusters: dict) -> tuple[list[tuple[str, str]], list[int]]:
    '''
    Generate pairs of entities and labels for the siamese network
    
    Args:
        entity2clusters (dict): dictionary containing clusters of entities
        
    Returns:
        pairs (list[tuple[str, str]]): list of pairs of entities
        labels (list[int]): list of labels
    '''
    pairs = []
    labels = []

    positive_pairs = 0
    negative_pairs = 0
    
    for entity_id, clusters in entity2clusters.items():
        
        # Generate positive pairs
        for _, items in clusters.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)): 
                    pairs.append((items[i], items[j]))
                    labels.append(1)
                    positive_pairs += 1

        # Generate negative pairs     
        for other_entity_id, other_clusters in entity2clusters.items():
            if other_entity_id == entity_id:
                continue
            
            for _, other_items in other_clusters.items():
                for item in items:
                    for other_item in other_items:
                        pairs.append((item, other_item))
                        labels.append(0)
                        negative_pairs += 1

    print(f"Generated {len(pairs)} pairs: {positive_pairs} positive pairs and {negative_pairs} negative pairs")
    
    return pairs, labels


if __name__ == "__main__":
    entity2clusters = json.load(open(paths.RESULTS_DIR + "/evaluation/entity2clusters.json"))
    embeddings = json.load(open(paths.RESULTS_DIR + "/embeddings/embeddings_distilbert_base_uncased_preprocessed.json"))

    parser = argparse.ArgumentParser(description="Process a directory of JSON files to extract and embed page titles.")
    parser.add_argument("--load_epoch", type=int, help="Load the model and optimizer state from the specified epoch to continue training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model (default: 10)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Computing device to use (cpu, cuda, or mps)")
    args = parser.parse_args()
    
    # Determine the device to use
    device = assign_device(args.device)

    model = SiameseNetwork(768, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    if args.load_epoch is not None:
        try:
            model.load_state_dict(torch.load(paths.MODELS_DIR + f"/siamese_net/siamese_model_epoch_{args.load_epoch}.pth"))
            optimizer.load_state_dict(torch.load(paths.MODELS_DIR + f"/siamese_net/siamese_optimizer_epoch_{args.load_epoch}.pth"))
            print(f"\033[32mModel and optimizer loaded from epoch {args.load_epoch}\033[0m")
        except Exception as e:
            print("\033[31m!!! --- Error loading model and optimizer states. Make sure model and optimizer state files exist in model directory --- !!!\033[0m")
            print(f"Error details: {e}")
            sys.exit(1)
    
    pairs, labels = generate_pairs(entity2clusters)

    dataset = SiameseDataset(pairs, labels, embeddings)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()  # Set the model to training mode (only necessary if you had previously set it to eval mode)
    model.to(device) # Move the model to the device
    
    criterion = ContrastiveLoss()

    training = SiameseTraining(model, dataloader, criterion, optimizer)
    training.train(epochs=args.num_epochs, print_every=1000)

    # Save the final trained model and optimizer state
    torch.save(model.state_dict(), paths.MODELS_DIR + "/siamese_net/siamese_model_final.pth")
    torch.save(optimizer.state_dict(), paths.MODELS_DIR + "/siamese_net/siamese_optimizer_final.pth")

    print("\033[32mFinal model and optimizer states have been saved.\033[0m")



