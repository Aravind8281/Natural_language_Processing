from transformers import pipeline
text="I love huggig face transformer"
classifier=pipeline("sentiment-analysis")
print(classifier(text))
