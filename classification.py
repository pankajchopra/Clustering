from transformers import pipeline

# Load a zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Example input and labels
input_text = "I want to book a flight from New York to Los Angeles."
candidate_labels = ["book flight", "order food", "cancel reservation"]

# Zero-shot classification
result = classifier(input_text, candidate_labels)
print(result)