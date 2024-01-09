from transformers import pipeline

# Load multitask model for text classification and named entity recognition
multitask_classifier = pipeline(
    task="text-classification",
    model="a-username/a-model-name",  # Replace with an actual model name or path
)

# Example: Multitask learning on a text
text = "Hugging Face Transformers is a powerful library for NLP tasks."

# Perform multitask classification
result = multitask_classifier(text)

# Print the result
print(result)
