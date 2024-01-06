from textblob import TextBlob

text = "TextBlob is a powerful library for processing textual data."
blob = TextBlob(text)

print("Tags:", blob.tags)
