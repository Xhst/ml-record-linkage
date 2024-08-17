import argparse
from itertools import combinations
import random
import json
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import paths
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass in a single network

        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            x (torch.Tensor): output tensor
        '''
        x = self.input_layer(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.hidden_layer1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        
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
    def __init__(self, margin=0.1):
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
        try:
            pair = (self.embeddings[self.pairs[idx][0]], self.embeddings[self.pairs[idx][1]])
        except KeyError as e:
            print(f"KeyError: {e} not found in embeddings")
            # Handle the error or provide a default value
            # For example, return a tuple of zeros if a key is missing
            pair = (torch.zeros_like(next(iter(self.embeddings.values()))), torch.zeros_like(next(iter(self.embeddings.values()))))
        return pair, self.labels[idx]
    

class SiameseTraining:
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        

    def train(self, current_epoch: int, epochs: int, enable_prints: bool = True, print_every: int = 10):
        total_loss = 0.0
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
            
            total_loss += loss.item()
            
            if enable_prints and i % print_every == 0:
                print(f'Epoch [{current_epoch + 1}/{epochs}], Item [{i}/{len(self.dataloader)}], Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.dataloader)
        
        scheduler.step(avg_loss)
        
        print(f'\033[32mEpoch [{current_epoch + 1}/{epochs}], Avg Loss: {avg_loss:.6f}\033[0m')


def load_pairs_from_file(file_path: str) -> tuple[list[tuple[str, str]], list[int]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['pairs'], data['labels']


def save_pairs_to_file(pairs: list[tuple[str, str]], labels: list[int], file_path: str):
    with open(file_path, 'w') as f:
        json.dump({'pairs': pairs, 'labels': labels}, f, indent=4)


def filter_negative_pairs(pairs: list[tuple[str, str]], labels: list[int], neg_pos_ratio: float) -> tuple[list[tuple[str, str]], list[int]]:
    '''
    Filter negative pairs based on a specified ratio, keeping all positive pairs.
    
    Args:
        pairs (list[tuple[str, str]]): List of pairs of entities.
        labels (list[int]): List of labels (1 for positive, 0 for negative).
        neg_pos_ratio (float): Ratio of negative to positive pairs to keep.
        
    Returns:
        filtered_pairs (list[tuple[str, str]]): Filtered list of pairs.
        filtered_labels (list[int]): Labels corresponding to the filtered pairs.
    '''
    # Separate positive and negative pairs
    positive_pairs = [pair for pair, label in zip(pairs, labels) if label == 1]
    negative_pairs = [pair for pair, label in zip(pairs, labels) if label == 0]

    # Calculate the number of negative pairs to keep
    num_positive = len(positive_pairs)
    num_negative = int(num_positive * neg_pos_ratio)
    
    # Randomly sample the negative pairs
    sampled_negative_pairs = random.sample(negative_pairs, min(num_negative, len(negative_pairs)))
    
    # Combine the positive pairs with the sampled negative pairs
    filtered_pairs = positive_pairs + sampled_negative_pairs
    filtered_labels = [1] * len(positive_pairs) + [0] * len(sampled_negative_pairs)
    
    # Shuffle the combined dataset
    combined = list(zip(filtered_pairs, filtered_labels))
    random.shuffle(combined)
    
    filtered_pairs, filtered_labels = zip(*combined)
    
    return list(filtered_pairs), list(filtered_labels)


def split_data(pos_pairs: list[tuple[str, str]], neg_pairs: list[tuple[str, str]], test_size=0.2) -> tuple[list[tuple[str, str]], list[int], list[tuple[str, str]], list[int]]:
    # Split positive pairs
    pos_train, pos_test = train_test_split(pos_pairs, test_size=test_size, random_state=42)
    # Split negative pairs
    neg_train, neg_test = train_test_split(neg_pairs, test_size=test_size, random_state=42)
    
    # Combine
    train_pairs = pos_train + neg_train
    test_pairs = pos_test + neg_test
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)
    test_labels = [1] * len(pos_test) + [0] * len(neg_test)
    
    # Shuffle
    combined_train = list(zip(train_pairs, train_labels))
    combined_test = list(zip(test_pairs, test_labels))
    random.shuffle(combined_train)
    random.shuffle(combined_test)
    
    train_pairs, train_labels = zip(*combined_train)
    test_pairs, test_labels = zip(*combined_test)
    
    return list(train_pairs), list(train_labels), list(test_pairs), list(test_labels)


def generate_pairs(file_path: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    '''
    Generate positive (similar) and negative pairs of entities for the siamese
    network from the ground truth csv.
    
    Args:
        file_path (str): path to the ground truth CSV file
        
    Returns:
        positive_pairs (list[tuple[str, str]]): list of positive pairs of entities
        negative_pairs (list[tuple[str, str]]): list of negative pairs of entities
    '''
    
    # Load data into a pandas DataFrame from a CSV file
    df = pd.read_csv(file_path, header=None, names=['Entity', 'URL'], skiprows=1)
    
    # Create a dictionary to store URLs by entity
    entity_urls = df.groupby('Entity')['URL'].apply(list).to_dict()
    
    positive_pairs = []
    negative_pairs = []
    
    # Generate positive pairs (same entity)
    for urls in entity_urls.values():
        for pair in combinations(urls, 2):
            positive_pairs.append(pair)
    
    # Generate negative pairs (different entities)
    for (url1, entity1), (url2, entity2) in combinations(df[['URL', 'Entity']].itertuples(index=False), 2):
        if entity1 != entity2:
            negative_pairs.append((url1, url2))
    
    print(f"Generated {len(positive_pairs) + len(negative_pairs)} pairs: {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
    
    return list(positive_pairs), list(negative_pairs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a siamese network for pairwise entity matching")
    parser.add_argument("--load_epoch", type=int, help="Load the model and optimizer state from the specified epoch to continue training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model (default: 10)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Computing device to use (cpu, cuda, or mps)")
    parser.add_argument("--train_data", type=str, help="Path to training data JSON file")
    parser.add_argument("--pre", type=str, choices=["y","n","N","Y"], default="y", help="'y/Y' if you want to train with the preprocessed embeddings, 'n/N' otherwise")
    args = parser.parse_args()
    
    # Determine the device to use
    device = assign_device(args.device)

    model = SiameseNetwork(768, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # Load the embeddings
    if args.pre.lower() == "n":
        embeddings = json.load(open(paths.RESULTS_DIR + "/embeddings/embeddings_distilbert_base_uncased.json"))
        print("\033[33mLoaded raw embeddings\033[0m")
    else:
        embeddings = json.load(open(paths.RESULTS_DIR + "/embeddings/embeddings_distilbert_base_uncased_preprocessed.json"))
        print("\033[33mLoaded preprocessed embeddings\033[0m")
    
    # Load the model and optimizer states if specified
    if args.load_epoch is not None:
        try:
            model.load_state_dict(torch.load(paths.MODELS_DIR + f"/siamese_net/siamese_model_epoch_{args.load_epoch}.pth"))
            optimizer.load_state_dict(torch.load(paths.MODELS_DIR + f"/siamese_net/siamese_optimizer_epoch_{args.load_epoch}.pth"))
            print(f"\033[32mModel and optimizer loaded from epoch {args.load_epoch}\033[0m")
        except Exception as e:
            print("\033[31m!!! --- Error loading model and optimizer states. Make sure model and optimizer state files exist in model directory --- !!!\033[0m")
            print(f"Error details: {e}")
            sys.exit(1)

    # Load training and test data or create it
    if args.train_data:
        # Load pre-saved pairs and labels
        train_pairs, train_labels = load_pairs_from_file(args.train_data)
    else:
        # Generate and save pairs
        pos_pairs, neg_pairs = generate_pairs(paths.GROUND_TRUTH_DIR + "/monitor_entity_resolution_gt(in).csv")
        train_pairs, train_labels, test_pairs, test_labels = split_data(pos_pairs, neg_pairs, test_size=0.2)
        pairs_output_dir = paths.DATASET_DIR + "/../siamese_splits/"
        save_pairs_to_file(train_pairs, train_labels, pairs_output_dir + 'train_pairs.json')
        save_pairs_to_file(test_pairs, test_labels, pairs_output_dir + 'test_pairs.json')
    
    
    model.train()  # Set the model to training mode (only necessary if you had previously set it to eval mode)
    model.to(device) # Move the model to the device
    
    criterion = ContrastiveLoss()
    
    filt_pairs, filt_labels = filter_negative_pairs(train_pairs, train_labels, neg_pos_ratio=2.0)

    train_dataset = SiameseDataset(filt_pairs, filt_labels, embeddings)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train the model for this epoch
    training = SiameseTraining(model, train_dataloader, criterion, optimizer)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    save_every = 5
    
    for epoch in range(args.num_epochs):
        # Generate fresh pairs for the current epoch
        
        training.train(current_epoch=epoch, epochs=args.num_epochs, enable_prints=False, print_every=100)  # Train for 1 epoch
        
        if epoch % save_every == 0:
            # Save the model at the end of each epoch
            torch.save(model.state_dict(), paths.MODELS_DIR + f"/siamese_net/siamese_model_epoch_{epoch}.pth")
            torch.save(optimizer.state_dict(), paths.MODELS_DIR + f"/siamese_net/siamese_optimizer_epoch_{epoch}.pth")
            print(f"\033[32mSaved model and optimizer state for epoch {epoch}\033[0m")
        
        # Generate new pairs for the next epoch
        filt_pairs, filt_labels = filter_negative_pairs(train_pairs, train_labels, neg_pos_ratio=2.0)

        train_dataset = SiameseDataset(filt_pairs, filt_labels, embeddings)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        training.dataloader = train_dataloader

    # Save the final trained model and optimizer state
    torch.save(model.state_dict(), paths.MODELS_DIR + "/siamese_net/siamese_model_final.pth")
    torch.save(optimizer.state_dict(), paths.MODELS_DIR + "/siamese_net/siamese_optimizer_final.pth")

    print("\033[32mFinal model and optimizer states have been saved.\033[0m")



