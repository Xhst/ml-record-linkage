import argparse
import os
import json
from transformers import AutoTokenizer, AutoModel
import torch


# Initialize the model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed_text(text, device):
    """Embed the given text using a pre-trained transformer model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the token embeddings as the sentence embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    # Convert the embedding to a list
    embedding_list = embedding.tolist()
    return embedding_list


def process_json_file(file_path, device):
    """Extract the page title from the JSON file and return its embedding."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        page_title = data.get("<page title>", "")
        if page_title:
            return embed_text(page_title, device)
    return None


def process_directory(directory_path, device):
    """Process all JSON files in the directory and return a list of embeddings."""
    filepath2embedding = {}
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                embedding = process_json_file(file_path, device)
                if embedding is not None:
                    relative_path = os.path.relpath(file_path, directory_path)
                    filepath2embedding[relative_path] = embedding
    return filepath2embedding


def count_files(directory_path):
    """Count the number of JSON files in the directory."""
    count = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                count += 1
    return count


def assign_device(device_name):
    torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU.")
        else:
            print("CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    
    # Display available devices
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            

def save_embedding_dictionary(filepath2embedding, output_dir, output_file_name):
    """Save the embedding dictionary to a JSON file in the specified directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Define the file path for the JSON file
    json_file_path = os.path.join(output_dir, output_file_name)
    # Write the dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(filepath2embedding, json_file, indent=4)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a directory of JSON files to extract and embed page titles.")
    parser.add_argument("directory", type=str, help="Path to the directory containing JSON files")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Computing device to use (cpu or cuda)")
    args = parser.parse_args()

    # Determine the device to use
    device = assign_device(args.device)

    # Process the specified directory into a dictionary of embeddings
    embeddings = process_directory(args.directory, device)
    
    print("Embedded " + str(len(embeddings)) + " jsons out of " + str(count_files(args.directory)) + " total. \n" + 
          str(count_files(args.directory) - len(embeddings)) + " missing") # 16627
    
    print("Saving to json...")
    
    output_dir = "embeddings"
    save_embedding_dictionary(embeddings, output_dir, "embeddings_distilbert_base_uncased.json")
    
    print("Embeddings saved to " + output_dir + "/embeddings_distilbert_base_uncased.json")


if __name__ == "__main__":
    main()
