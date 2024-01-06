from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
sentences = [
    "Word embeddings are powerful tools in NLP.",
    "They capture semantic relationships between words.",
    "Word2Vec is a popular technique for creating word embeddings."
]
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['word']
similar_words = model.wv.most_similar('word', topn=3)
print("Vector for 'word':", vector)
print("Similar words to 'word':", similar_words)
