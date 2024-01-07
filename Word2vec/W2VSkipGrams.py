from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize 
import nltk
corpus = "Skip-gram is a type of Word2Vec model used for natural language processing tasks."
tokenized_corpus = word_tokenize(corpus.lower())
model = Word2Vec(sentences=[tokenized_corpus], vector_size=100, window=5, sg=1, min_count=1)
vector_representation = model.wv['skip-gram']
similar_words = model.wv.most_similar('skip-gram', topn=5)
print("Vector representation of 'skip-gram':", vector_representation)
print("Similar words to 'skip-gram':", similar_words)
