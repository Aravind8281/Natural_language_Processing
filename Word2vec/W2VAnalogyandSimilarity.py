from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

path = api.load('word2vec-google-news-300', return_path=True)
model = KeyedVectors.load_word2vec_format(path, binary=True)

def similarity(word1, word2):
    similarity = model.similarity(word1, word2)
    print("Similarity:", similarity)

def analogy(positive, negative, topn=1):
    analogy_result = model.most_similar(positive=positive, negative=negative, topn=topn)
    print("Analogy:", analogy_result[0][0])

similarity("king", "queen")
analogy(['man', 'queen'], ['woman'])
