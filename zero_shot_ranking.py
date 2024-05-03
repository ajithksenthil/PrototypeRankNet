from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

def embed_text(texts):
    """ Embeds a list of texts and returns the average embedding. """
    all_embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            all_embeddings.append(outputs.last_hidden_state.mean(dim=1))
    # Calculate the average across all embeddings
    return torch.mean(torch.stack(all_embeddings), dim=0)

# Define lists of anchor texts for high and low optimism
texts_high = [
    "The company has had a tremendously successful year, and future prospects look excellent.",
    "Unprecedented growth and profitability have been achieved this quarter.",
    "Record-breaking achievements and positive market feedback."
]

texts_low = [
    "The company has faced significant challenges and uncertainties, with no clear resolution in sight.",
    "Declining sales and poor customer feedback continue to trouble the business.",
    "Market conditions have been difficult and are expected to worsen."
]

# Compute average embeddings for both high and low optimism texts
embed_high = embed_text(texts_high)
embed_low = embed_text(texts_low)

# Calculate the direction vector using the average embeddings
direction_vector = embed_high - embed_low

def rank_texts_by_optimism(texts):
    """ Ranks given texts by optimism based on their projection onto the direction vector. """
    projections = [(text, (torch.dot(embed_text([text]).flatten(), direction_vector.flatten()) / torch.norm(direction_vector)).item()) for text in texts]
    return sorted(projections, key=lambda x: x[1], reverse=True)

# Define a list of texts to be ranked
texts = [
    "The company is expected to perform well in the next quarter.",
    "There is a high likelihood of surpassing all previous sales records.",
    "Revenue has declined over the past year.",
    "Things are looking amazing right now, could not be better",
    "Market conditions have been difficult, but there's potential for recovery.",
    "The economic outlook is grim and likely to worsen."
]

# Rank the texts by optimism and print the results
ranked_texts = rank_texts_by_optimism(texts)
for text, score in ranked_texts:
    print(f"Score: {score:.2f} | Text: {text}")

# Plotting the projections
projections = [projection for _, projection in ranked_texts]
y_values = range(len(texts))

plt.figure(figsize=(10, 8))
plt.scatter(projections, y_values, color='blue')
for i, txt in enumerate(texts):
    plt.annotate(txt, (projections[i], y_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.axvline(x=0, color='r', linestyle='--')
plt.title('Projection of Texts onto the Semantic Direction of Optimism')
plt.xlabel('Projection Magnitude (Alignment with Optimism)')
plt.ylabel('Texts (Indexed)')
plt.yticks(ticks=y_values, labels=[f'Text {i+1}' for i in y_values])
plt.grid(True)
plt.show()
