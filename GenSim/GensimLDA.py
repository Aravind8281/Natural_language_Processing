from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
documents = [
    "Topic modeling is an interesting area of natural language processing.",
    "Latent Dirichlet Allocation is a popular technique for topic modeling.",
    "Gensim provides an implementation of Latent Dirichlet Allocation.",
    "The Python programming language is commonly used in data science.",
    "Stopwords are common words that are often removed in text processing.",
]
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens
processed_documents = [preprocess_text(doc) for doc in documents]
dictionary = corpora.Dictionary(processed_documents)
corpus = [dictionary.doc2bow(doc) for doc in processed_documents]
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
