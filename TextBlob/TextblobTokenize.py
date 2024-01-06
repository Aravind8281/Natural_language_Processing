from textblob import TextBlob
text = "TextBlob is a powerful library for processing textual data."
blob = TextBlob(text)
print("Sentences:", blob.sentences)
print("Words:", blob.words)
print("Tags:", blob.tags)
