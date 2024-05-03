from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load the tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Function to embed a list of texts and return average embedding
def embed_text(texts):
    all_embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            all_embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.mean(torch.stack(all_embeddings), dim=0)

# Dictionary of optimism levels with example texts
optimism_levels = {
    'very_low': [
        "There appears to be no solution on the horizon for the myriad of problems we face.",
        "With continued losses and no strategic direction, the future looks bleak.",
        "The company has reached a critical point and may not survive the next quarter.",
        "Investors are losing hope as the situation worsens with no end in sight.",
        "The downturn has been long and recovery seems increasingly unlikely."
    ],
    'low': [
        "Recent developments have been disappointing, and improvement is slow.",
        "Challenges in the market continue to suppress any significant progress.",
        "Growth has stalled, and the outlook remains uncertain at best.",
        "The results have been underwhelming, with few signs of a turnaround.",
        "Concerns are growing as the company struggles to meet its goals."
    ],
    'medium': [
        "The company is holding steady, with potential for some growth.",
        "Performance is average, mirroring industry trends without clear leads.",
        "There are as many opportunities as there are challenges ahead.",
        "Results are mixed, with some positive developments and some setbacks.",
        "The situation is stable, but significant growth is not expected soon."
    ],
    'high': [
        "There is good momentum, and the next quarter looks promising.",
        "The company has overcome several obstacles and is poised for growth.",
        "Positive trends are emerging, suggesting better times ahead.",
        "The team's efforts are paying off, leading to optimistic forecasts.",
        "Improvements in key areas are likely to boost overall performance soon."
    ],
    'very_high': [
        "Expectations are high as the company continues to exceed every target.",
        "With innovation at its peak, the company is set to dominate the market.",
        "Record profits are anticipated following a series of successful ventures.",
        "The company is at the forefront of a major breakthrough in the industry.",
        "Unprecedented success is imminent, thanks to groundbreaking strategies."
    ]
}


# Compute embeddings for each optimism level
prototypes = {level: embed_text(texts) for level, texts in optimism_levels.items()}

# Compute direction vectors between each level
def compute_interlevel_direction_vectors(prototypes):
    levels = list(prototypes.keys())
    direction_vectors = []
    for i in range(len(levels) - 1):
        vector = prototypes[levels[i + 1]] - prototypes[levels[i]]
        normalized_vector = vector / torch.norm(vector)
        direction_vectors.append(normalized_vector)
    return direction_vectors

# Aggregate direction vectors
def aggregate_direction_vectors(direction_vectors):
    aggregated_vector = torch.mean(torch.stack(direction_vectors), dim=0)
    return aggregated_vector

direction_vectors = compute_interlevel_direction_vectors(prototypes)
aggregated_vector = aggregate_direction_vectors(direction_vectors)

# Function to project text on the aggregated vector
def project_text_on_aggregated_vector(text):
    text_embedding = embed_text([text])
    projection = torch.dot(text_embedding.flatten(), aggregated_vector.flatten()) / torch.norm(aggregated_vector)
    return projection.item()

# Example usage: Projecting texts and sorting by optimism
texts = [
    "The company is expected to perform well in the next quarter.",
    "There is a high likelihood of surpassing all previous sales records.",
    "Revenue has declined over the past year.",
    "Things are looking amazing right now, could not be better",
    "Market conditions have been difficult, but there's potential for recovery.",
    "The economic outlook is grim and likely to worsen."
]

ranked_texts = sorted([(text, project_text_on_aggregated_vector(text)) for text in texts], key=lambda x: x[1], reverse=True)
for text, score in ranked_texts:
    print(f"Score: {score:.2f} | Text: {text}")
