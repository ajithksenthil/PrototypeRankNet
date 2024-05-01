import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

class OptimismProtoNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

    def forward(self, x, prototypes):
        transformed_x = self.transform(x)
        # Transform prototypes once per forward pass, consider caching if unchanging
        transformed_prototypes = torch.stack([self.transform(proto.unsqueeze(0)).squeeze(0) for proto in prototypes])
        return torch.cdist(transformed_x, transformed_prototypes)

def get_contextual_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def compute_prototypes(samples_by_level):
    prototypes = {}
    for level, samples in samples_by_level.items():
        embeddings = torch.stack([get_contextual_embedding(text) for text in samples])
        prototypes[level] = embeddings.mean(dim=0)
    return prototypes

optimism_levels = {
    'very_low': ["The company is struggling severely with no signs of improvement."],
    'low': ["There are some concerns about the next quarter."],
    'medium': ["The company is doing okay but could be better."],
    'high': ["The company's prospects are looking good!"],
    'very_high': ["We are expecting record-breaking results this year!"]
}

prototypes = compute_prototypes(optimism_levels)

# Create and initialize model and optimizer once
proto_net = OptimismProtoNet(768, 512, 0.5)
optimizer = optim.Adam(proto_net.parameters(), lr=0.001)

# Add your training, validation, and other functional code here following similar optimization principles
def train_optimism_proto_net(proto_net, optimizer, episode_data, prototypes):
    proto_net.train()
    total_loss = 0
    for episode in episode_data:
        optimizer.zero_grad()  # Ensure gradient buffers are reset

        texts, true_level = episode_data[episode]
        embeddings = torch.stack([get_contextual_embedding(text) for text in texts])

        # Ensure prototype tensors are detached to avoid any graph retention issues
        prototype_tensors = torch.stack([prototypes[level].detach() for level in prototypes])

        dists = proto_net(embeddings, prototype_tensors)
        labels = torch.tensor([list(prototypes.keys()).index(true_level) for _ in texts], dtype=torch.long)

        loss = F.cross_entropy(-dists, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(texts)
        print(f"Loss calculated: {loss.item()}, Embeddings shape: {embeddings.shape}")

    avg_loss = total_loss / sum(len(texts) for _, texts in episode_data.items())
    return avg_loss

# def train_optimism_proto_net(proto_net, optimizer, training_data, prototypes):
#     proto_net.train()  # Set the model to training mode
#     total_loss = 0
#     total_samples = 0
    
#     for texts, true_level in training_data:
#         optimizer.zero_grad()  # Reset gradients
        
#         # Transform texts to embeddings
#         embeddings = torch.stack([get_contextual_embedding(text) for text in texts])
        
#         # Get the transformed prototypes (ensure they're detached to prevent gradient updates)
#         prototype_tensors = torch.stack([prototypes[level].detach() for level in prototypes])
        
#         # Compute distances
#         dists = proto_net(embeddings, prototype_tensors)
        
#         # Prepare labels for loss calculation
#         labels = torch.tensor([list(prototypes.keys()).index(true_level) for _ in texts], dtype=torch.long)
        
#         # Compute loss
#         loss = F.cross_entropy(-dists, labels)  # Negative distances because closer means more similar
#         loss.backward()  # Compute gradients
#         optimizer.step()  # Update weights
        
#         total_loss += loss.item() * len(texts)
#         total_samples += len(texts)
    
#     avg_loss = total_loss / total_samples
#     print(f"Training Loss: {avg_loss:.4f}")
#     return avg_loss

def validate_optimism_proto_net(proto_net, validation_data, prototypes):
    proto_net.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for label, (texts, true_level) in validation_data.items():
            embeddings = torch.stack([get_contextual_embedding(text) for text in texts])
            dists = proto_net(embeddings, torch.stack(list(prototypes.values())))

            labels = torch.tensor([list(prototypes.keys()).index(true_level) for _ in texts], dtype=torch.long)
            loss = F.cross_entropy(-dists, labels)
            total_loss += loss.item() * len(texts)

            _, predicted = torch.min(dists, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(texts)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


# def validate_optimism_proto_net(proto_net, validation_data, prototypes):
#     proto_net.eval()  # Set the model to evaluation mode
#     total_loss = 0
#     total_correct = 0
#     total_samples = 0

#     with torch.no_grad():  # No gradient tracking needed
#         for texts, true_level in validation_data:
#             embeddings = torch.stack([get_contextual_embedding(text) for text in texts])
#             prototype_tensors = torch.stack(list(prototypes.values()))

#             dists = proto_net(embeddings, prototype_tensors)
#             labels = torch.tensor([list(prototypes.keys()).index(true_level) for _ in texts], dtype=torch.long)
            
#             loss = F.cross_entropy(-dists, labels)
#             total_loss += loss.item() * len(texts)
            
#             _, predicted = torch.min(dists, dim=1)
#             total_correct += (predicted == labels).sum().item()
#             total_samples += len(texts)

#         avg_loss = total_loss / total_samples
#         accuracy = total_correct / total_samples * 100
#         print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
#         return avg_loss, accuracy


def compute_direction_vector(prototypes):
    # Correctly order the keys based on actual optimism levels
    vector_keys = ['very_low', 'low', 'medium', 'high', 'very_high']
    direction_vector = prototypes[vector_keys[-1]] - prototypes[vector_keys[0]]
    return direction_vector




# Example training and validation episodes
train_episode_data = {
    'episode1': (["Company profits are soaring beyond forecasts.", "Modest gains are expected."], 'high'),
    'episode2': (["There are looming risks that might affect the stability.", "Uncertainty clouds the fiscal projections."], 'low')
}

validation_data = {
    'validation1': (["There is stability in the financial outlook.", "Slight downturns are anticipated."], 'medium'),
    'validation2': (["Optimism is thriving with strong market leadership.", "Potential setbacks are manageable."], 'high')
}

# Training loop
input_size = 768  # For bert-base-uncased
hidden_size = 512  # Example hidden size
dropout_rate = 0.5  # Dropout rate

proto_net = OptimismProtoNet(input_size, hidden_size, dropout_rate)
optimizer = optim.Adam(proto_net.parameters(), lr=0.001)

for episode in train_episode_data:
    loss = train_optimism_proto_net(proto_net, optimizer, {episode: train_episode_data[episode]}, prototypes)
    print(f'Training Loss: {loss:.4f}')

# Validation
avg_loss, accuracy = validate_optimism_proto_net(proto_net, validation_data, prototypes)
print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')



def compute_comprehensive_direction_vector(prototypes):
    # Collect all prototypes into a matrix
    prototype_matrix = torch.stack(list(prototypes.values()))
    
    # Perform PCA to find the principal component
    pca = PCA(n_components=1)  # We only need the first principal component to find the direction of optimism
    principal_components = pca.fit_transform(prototype_matrix.cpu().numpy())
    direction_vector = torch.tensor(principal_components[:, 0], dtype=torch.float32)
    
    # Normalize the direction vector
    direction_vector = direction_vector / torch.norm(direction_vector)
    return direction_vector

# Example usage:
prototypes = compute_prototypes(optimism_levels)  # Assuming this is already defined
direction_vector = compute_comprehensive_direction_vector(prototypes)
print("Comprehensive Direction Vector:", direction_vector)

def project_text_on_direction(text_embedding, direction_vector):
    # Flatten the embeddings and direction vector
    text_embedding_flat = text_embedding.flatten()
    direction_vector_flat = direction_vector.flatten()

    # Calculate the projection of the text embedding onto the direction vector
    projection = torch.dot(text_embedding_flat, direction_vector_flat)
    return projection

# Test with a sample text
text_embedding = get_contextual_embedding("The company's future is looking incredibly bright.")
projection = project_text_on_direction(text_embedding, direction_vector)
print("Projection of text on optimism direction:", projection.item())


def compute_interlevel_direction_vectors(prototypes):
    levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    direction_vectors = []
    
    for i in range(len(levels) - 1):
        vector = prototypes[levels[i + 1]] - prototypes[levels[i]]
        normalized_vector = vector / torch.norm(vector)
        direction_vectors.append(normalized_vector)
    
    return direction_vectors

def aggregate_direction_vectors(direction_vectors):
    aggregated_vector = torch.zeros_like(direction_vectors[0])
    for vector in direction_vectors:
        aggregated_vector += vector
    aggregated_vector = aggregated_vector / len(direction_vectors)  # Normalize
    return aggregated_vector

def project_text_on_aggregated_vector(text_embedding, aggregated_vector):
    text_embedding_flat = text_embedding.flatten()
    aggregated_vector_flat = aggregated_vector.flatten()
    projection = torch.dot(text_embedding_flat, aggregated_vector_flat)
    return projection

def score_texts(texts, proto_net, prototypes):
    embeddings = [get_contextual_embedding(text) for text in texts]
    direction_vectors = compute_interlevel_direction_vectors(prototypes)
    aggregated_vector = aggregate_direction_vectors(direction_vectors)
    
    projections = [project_text_on_aggregated_vector(embedding, aggregated_vector) for embedding in embeddings]
    scores = [(projection + 1) / 2 for projection in projections]  # Example of mapping to 0-1
    return scores

