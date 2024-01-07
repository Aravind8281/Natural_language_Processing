from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
sentence=[
    "Word embeddings are powerful tools in natural language processing.",
    "Word2Vec is a popular algorithm for generating word embeddings.",
    "It captures semantic relationships between words.",
    "Sentiment analysis benefits from using word embeddings."
]
token=[word_tokenize(word.lower()) for word in sentence]
model=Word2Vec(sentences=token,vector_size=100,window=5,min_count=1,workers=4)
model.save("model")
loaded=Word2Vec.load("model")
vector=loaded.wv['word']
print(vector) 
