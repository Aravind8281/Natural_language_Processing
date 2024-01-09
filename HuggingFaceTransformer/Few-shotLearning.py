from transformers import pipeline
classifier=pipeline("text-classification")
# Example: Few-shot classification with prompt engineering
prompt = "This is a few-shot classification task. Classify the following text:"
text = "A powerful tool for natural language processing tasks."

topics=["technology","science","sports"]
model=f"{prompt}{text}"
print(classifier(model,topics))
