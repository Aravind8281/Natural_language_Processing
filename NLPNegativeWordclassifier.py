import nltk.tokenize as token
from textblob import TextBlob
import nltk

nltk.download('punkt')

sentence = "Natural Language Processing (NLP) has revolutionized the way we interact with and analyze textual data. Its capabilities extend far beyond basic language understanding, allowing machines to comprehend, interpret, and respond to human language in a meaningful way. NLP algorithms are adept at extracting valuable insights from vast amounts of unstructured text, enabling applications such as sentiment analysis, text summarization, and named entity recognition. The continuous advancements in NLP techniques have paved the way for more accurate and context-aware language models. Researchers and developers alike are enthusiastic about the positive impact NLP can have on various industries, ranging from healthcare and finance to customer service and education. With its ability to unlock the hidden potential of linguistic data, NLP is contributing to the creation of smarter, more intuitive technologies that enhance our daily lives and drive innovation forward."

words = token.word_tokenize(sentence)
positive_words = []

for word in words:
    blob = TextBlob(word)
    if blob.sentiment.polarity < 0:
        positive_words.append(word)

print("Positive Words:", positive_words)
