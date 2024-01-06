from textblob import TextBlob
text = "I hate using TextBlob! It's a fantastic library."
blob = TextBlob(text)
sentiment = blob.sentiment
print("Sentiment Analysis:")
print(f"Polarity: {sentiment.polarity}")
print(f"Subjectivity: {sentiment.subjectivity}")
if sentiment.polarity > 0:
    print("Overall sentiment: Positive")
elif sentiment.polarity < 0:
    print("Overall sentiment: Negative")
else:
    print("Overall sentiment: Neutral")
