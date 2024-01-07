import gensim.models 
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Sample corpus
corpus = [
    "I love natural language processing.",
    "Word embeddings capture semantic relationships.",
    "CBOW is a technique in NLP.",
    "It learns from contextual words.",
    "Training a CBOW model is straightforward."
]

# Tokenize the corpus
tokenized_corpus = [simple_preprocess(sentence, deacc=True) for sentence in corpus]

# Create the Word2Vec model with negative sampling
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=2,
    sg=0,
    min_count=1,
    negative=15
)
target_word = "processing"
context_words = ["language", "natural"]
model.train([[target_word], context_words], epochs=model.epochs, total_examples=len(tokenized_corpus))
