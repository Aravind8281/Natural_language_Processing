from transformers import pipeline 
text="A Hugging face transformer is a best natural language processing tool"
classifier=pipeline("zero-shot-classification")
topics=["technology","science","sports"]
print(classifier(text,topics))
