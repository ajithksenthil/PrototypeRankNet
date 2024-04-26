from transformers import AutoTokenizer, AutoModel
import torch

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
