import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

class OptimismProtoNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(OptimismProtoNet, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

    def forward(self, x, prototypes):
        transformed_x = self.transform(x)
        transformed_prototypes = torch.stack([self.transform(proto.unsqueeze(0)).squeeze(0) for proto in prototypes])

        # Ensuring that both tensors are 2D and removing any extra dimensions
        transformed_x = transformed_x.squeeze(1) if transformed_x.dim() > 2 else transformed_x
        transformed_prototypes = transformed_prototypes.squeeze(1) if transformed_prototypes.dim() > 2 else transformed_prototypes

        print(f"transformed_x dimensions after adjustment: {transformed_x.shape}")
        print(f"transformed_prototypes dimensions after adjustment: {transformed_prototypes.shape}")

        return torch.cdist(transformed_x, transformed_prototypes)


def get_contextual_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Use pooling over tokens

# Function to compute prototypes
def compute_prototypes(samples_by_level):
    prototypes = {}
    for level, samples in samples_by_level.items():
        embeddings = torch.stack([get_contextual_embedding(text) for text in samples])
        prototypes[level] = embeddings.mean(dim=0)
    return prototypes

# Example of preparing optimism levels and their associated texts
optimism_levels = {
    'very_low': ["The company is struggling severely with no signs of improvement."],
    'low': ["There are some concerns about the next quarter."],
    'medium': ["The company is doing okay but could be better."],
    'high': ["The company's prospects are looking good!"],
    'very_high': ["We are expecting record-breaking results this year!"]
}

prototypes = compute_prototypes(optimism_levels)




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


# Initialize model and optimizer
input_size = 768  # Adjust as per model output
hidden_size = 512
dropout_rate = 0.5
proto_net = OptimismProtoNet(input_size, hidden_size, dropout_rate)
optimizer = optim.Adam(proto_net.parameters(), lr=0.001)

# Training and validation code should be adjusted according to the actual data and needs


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


# def rank_texts_by_optimism(texts, proto_net, prototypes):
#     embeddings = torch.stack([get_contextual_embedding(text) for text in texts])
#     dists = proto_net(embeddings, torch.stack(list(prototypes.values())))
#     rankings = torch.argmin(dists, dim=1)
#     return rankings

def rankings_to_scores(rankings, num_prototypes):
    # Assuming rankings are 0-indexed and should be mapped to a scale of 1 to num_prototypes
    scores = 1 + rankings  # This shifts the indices from 0-based to 1-based
    return scores


new_texts = ["The future looks bright with many possibilities.", "Risks are high and outlook is poor.", "Everything is falling apart", "I can't see how things could be any better", "We are looking at a bright future."]

def softmax_scores(distances):
    probabilities = F.softmax(-distances, dim=1)  # Apply softmax to negative distances
    scores = torch.arange(1, probabilities.shape[1] + 1)  # Scores from 1 to num_prototypes
    weighted_scores = torch.sum(probabilities * scores.float(), dim=1)
    return weighted_scores

# # Example of using this in validation
# embeddings = torch.stack([get_contextual_embedding(text) for text in new_texts])
# dists = proto_net(embeddings, torch.stack(list(prototypes.values())))
# softmax_scores = softmax_scores(dists)
# print("Softmax-based Optimism Scores:", softmax_scores)


# direction vector implementation

def compute_direction_vector(prototypes):
    # Assuming the prototypes dictionary is ordered from least to most optimistic
    vector_keys = sorted(prototypes.keys())
    direction_vector = prototypes[vector_keys[-1]] - prototypes[vector_keys[0]]
    return direction_vector

def project_text_on_direction(text_embedding, direction_vector):
    # Normalize the direction vector
    norm_direction_vector = direction_vector.flatten() / torch.norm(direction_vector)
    # Ensure text_embedding is also flattened
    text_embedding_flat = text_embedding.flatten()

    # Check dimensions
    print(f"Text embedding dimensions: {text_embedding_flat.shape}")
    print(f"Normalized direction vector dimensions: {norm_direction_vector.shape}")

    # Calculate projection
    projection = torch.dot(text_embedding_flat, norm_direction_vector)
    return projection


def rank_texts_by_optimism(texts, proto_net, prototypes):
    direction_vector = compute_direction_vector(prototypes)
    projections = []
    for text in texts:
        text_embedding = get_contextual_embedding(text)
        try:
            projection = project_text_on_direction(text_embedding, direction_vector)
            projections.append(projection.item())
        except RuntimeError as e:
            print(f"Error projecting text '{text}': {str(e)}")
            projections.append(None)  # Use None or a default value for failed projections
    return projections


# Initialize and train the model as before
for episode in train_episode_data:
    loss = train_optimism_proto_net(proto_net, optimizer, {episode: train_episode_data[episode]}, prototypes)
    print(f'Training Loss: {loss:.4f}')

    
print("prototypes: ", prototypes)
# Compute the direction vector from the learned prototypes
direction_vector = compute_direction_vector(prototypes)

# Rank new texts by their optimism based on the direction vector
new_texts_projections = rank_texts_by_optimism(new_texts, proto_net, prototypes)

# Print out the results
print("Text Optimism Projections:")
for text, projection in zip(new_texts, new_texts_projections):
    print(f"{text}: {projection:.2f}")

# Optionally, use softmax scores or normalized scores as before
