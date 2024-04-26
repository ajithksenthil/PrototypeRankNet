from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load the tokenizer and model from the Hugging Face Transformers library
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Ensure model is in evaluation mode
model.eval()


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Using mean pooling for simplicity

# Anchor texts
text_high = "The company has had a tremendously successful year, and future prospects look excellent."
text_low = "The company has faced significant challenges and uncertainties, with no clear resolution in sight."

# Compute embeddings
embed_high = embed_text(text_high)
embed_low = embed_text(text_low)

direction_vector = embed_high - embed_low

def rank_texts_by_optimism(texts):
    projections = []
    for text in texts:
        embed_text_vec = embed_text(text)
        projection = torch.dot(embed_text_vec.flatten(), direction_vector.flatten()) / torch.norm(direction_vector)
        projections.append((text, projection.item()))
    projections.sort(key=lambda x: x[1], reverse=True)  # Sorting by projection magnitude
    return projections

def get_projection_along_direction(texts, direction_vector):
    projections = []
    for text in texts:
        embed_text_vec = embed_text(text)
        # Project onto the semantic direction
        projection_magnitude = torch.dot(embed_text_vec.flatten(), direction_vector.flatten()) / torch.norm(direction_vector)
        projections.append(projection_magnitude.item())
    return projections



def get_projections_and_embeddings(texts):
    projections = []
    embeddings = []
    for text in texts:
        embed_text_vec = embed_text(text)
        projection = torch.dot(embed_text_vec.flatten(), direction_vector.flatten()) / torch.norm(direction_vector)
        projections.append(projection.item())
        embeddings.append(embed_text_vec.numpy())
    return np.array(projections), np.array(embeddings).reshape(-1, model.config.hidden_size)


texts = [
    "The company is expected to perform well in the next quarter.",
    "There is a high likelihood of surpassing all previous sales records.",
    "Revenue has declined over the past year.",
    "Things are looking amazing right now, could not be better",
    "Market conditions have been difficult, but there's potential for recovery.",
    "The economic outlook is grim and likely to worsen."
]

ranked_texts = rank_texts_by_optimism(texts)
for text, score in ranked_texts:
    print(f"Score: {score:.2f} | Text: {text}")

# projections, embeddings = get_projections_and_embeddings(texts)

projections = get_projection_along_direction(texts, direction_vector)

import matplotlib.pyplot as plt

# Simple y-values to offset texts vertically
y_values = range(len(texts))

plt.figure(figsize=(10, 8))
plt.scatter(projections, y_values, color='blue')

for i, txt in enumerate(texts):
    plt.annotate(txt, (projections[i], y_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.axvline(x=0, color='r', linestyle='--')
plt.title('Projection of Texts onto the Semantic Direction of Optimism')
plt.xlabel('Projection Magnitude (Alignment with Optimism)')
plt.ylabel('Texts (Indexed)')
plt.yticks(ticks=y_values, labels=[f'Text {i+1}' for i in y_values])  # Label each point with "Text 1", "Text 2", etc.
plt.grid(True)
plt.show()
