import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

class OptimismProtoNet(nn.Module):
    def __init__(self, input_size, hidden_size, initial_prototypes, dropout_rate=0.5):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.prototypes = nn.Parameter(initial_prototypes)

    def forward(self, x):
        transformed_x = self.transform(x)
        distances = torch.cdist(transformed_x, self.prototypes)
        return distances







# Tokenizer and model from Hugging Face's Transformers
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to get embeddings from BERT
def get_contextual_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Compute prototypes from text samples
def compute_prototypes(samples_by_level):
    prototypes = {}
    for level, samples in samples_by_level.items():
        embeddings = torch.stack([get_contextual_embedding(text) for text in samples])
        prototypes[level] = embeddings.mean(dim=0)
    return prototypes

# Prototypes based on predefined optimism levels
optimism_levels = {
    'very_low': ["The company is struggling severely with no signs of improvement."],
    'low': ["There are some concerns about the next quarter."],
    'medium': ["The company is doing okay but could be better."],
    'high': ["The company's prospects are looking good!"],
    'very_high': ["We are expecting record-breaking results this year!"]
}


# Compute direction vectors between each level
def compute_interlevel_direction_vectors(prototypes):
    levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    direction_vectors = []
    for i in range(len(levels) - 1):
        vector = prototypes[levels[i + 1]] - prototypes[levels[i]]
        normalized_vector = vector / torch.norm(vector)
        direction_vectors.append(normalized_vector)
    return direction_vectors

# Aggregate these direction vectors
def aggregate_direction_vectors(direction_vectors):
    aggregated_vector = torch.zeros_like(direction_vectors[0])
    for vector in direction_vectors:
        aggregated_vector += vector
    aggregated_vector = aggregated_vector / len(direction_vectors)
    return aggregated_vector

# Project text onto the aggregated direction vector
# def project_text_on_aggregated_vector(text_embedding, aggregated_vector):
#     text_embedding_flat = text_embedding.flatten()
#     aggregated_vector_flat = aggregated_vector.flatten()
#     projection = torch.dot(text_embedding_flat, aggregated_vector_flat)
#     return projection
def project_text_on_aggregated_vector(text_embedding, aggregated_vector, transform_layers):
    # Transform the text embedding to match the prototype dimensions
    transformed_text_embedding = transform_layers(text_embedding.unsqueeze(0)).squeeze(0)
    # Flatten embeddings for dot product
    transformed_text_embedding_flat = transformed_text_embedding.flatten()
    aggregated_vector_flat = aggregated_vector.flatten()
    # Compute projection
    projection = torch.dot(transformed_text_embedding_flat, aggregated_vector_flat)
    return projection



# Score texts based on their projections
def score_texts(texts, model, aggregated_vector, transform):
    embeddings = [get_contextual_embedding(text) for text in texts]
    projections = [project_text_on_aggregated_vector(embedding, aggregated_vector, transform) for embedding in embeddings]
    scores = normalize_scores(projections)
    return scores



# After training, compute direction vectors
def compute_direction_vectors(proto_net):
    num_prototypes = proto_net.prototypes.shape[0]
    direction_vectors = []
    for i in range(num_prototypes - 1):
        vector = proto_net.prototypes[i + 1] - proto_net.prototypes[i]
        normalized_vector = vector / torch.norm(vector)
        direction_vectors.append(normalized_vector)
    return direction_vectors




def compute_initial_prototypes(data):
    prototypes = {}
    for level, texts in data.items():
        embeddings = torch.stack([get_contextual_embedding(text) for text in texts])
        mean_embedding = embeddings.mean(dim=0)
        prototypes[level] = mean_embedding
    return prototypes

def transform_prototypes(prototypes, transform_layers):
    transformed_prototypes = []
    for proto in prototypes:
        transformed_proto = transform_layers(proto.unsqueeze(0)).squeeze(0)  # Apply the same transformations as in the model
        transformed_prototypes.append(transformed_proto)
    return torch.stack(transformed_prototypes)



# Function to compute and transform initial prototypes
def compute_and_transform_prototypes(data, transform):
    prototypes = {}
    for level, texts in data.items():
        embeddings = torch.stack([get_contextual_embedding(text) for text in texts])
        mean_embedding = embeddings.mean(dim=0)
        transformed_embedding = transform(mean_embedding.unsqueeze(0)).squeeze(0)
        prototypes[level] = transformed_embedding
    return torch.stack(list(prototypes.values()))


# Precompute embeddings for each level
optimism_embeddings = {level: torch.stack([get_contextual_embedding(text) for text in texts]) for level, texts in optimism_levels.items()}



def train_model(model, data, optimizer, epochs=10):
    for epoch in range(epochs):
        for level, embeddings in data.items():
            if embeddings.size(0) == 1:
                continue  # Skip if there is only one embedding to avoid dimension issues
            optimizer.zero_grad()
            distances = model(embeddings)
            distances = distances.squeeze()  # Correct shape if necessary
            prototype_idx = list(data.keys()).index(level)
            target = torch.full((embeddings.size(0),), prototype_idx, dtype=torch.long)

            if distances.shape[0] != target.shape[0]:
                raise ValueError(f"Mismatched shapes for distances ({distances.shape[0]}) and target ({target.shape[0]}) tensors")

            loss = F.cross_entropy(-distances, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}')








# Main code

# Instantiate and train model

proto_net = OptimismProtoNet(input_size=768, hidden_size=512, initial_prototypes=torch.randn(5, 256), dropout_rate=0.5)
# Assuming compute_and_transform_prototypes function correctly handles ordering
initial_prototype_tensor = compute_and_transform_prototypes(optimism_levels, proto_net.transform)
proto_net.prototypes = nn.Parameter(initial_prototype_tensor)

optimizer = optim.Adam(proto_net.parameters(), lr=0.001)



train_model(proto_net, optimism_embeddings, optimizer)



def normalize_scores(projections):
    min_proj, max_proj = min(projections), max(projections)
    return [(proj - min_proj) / (max_proj - min_proj) for proj in projections]

# Example usage
texts_to_score = ["The company's future looks promising.", "Risks are looming on the horizon."]
direction_vectors = compute_direction_vectors(proto_net)
aggregated_vector = aggregate_direction_vectors(direction_vectors)
scores = score_texts(texts_to_score, proto_net, aggregated_vector, proto_net.transform)

# Output results
print("Text Scores:")
for text, score in zip(texts_to_score, scores):
    print(f"Text: {text}, Score: {score:.2f}")