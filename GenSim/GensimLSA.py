from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LsiModel
from pprint import pprint
documents = [
    "Gensim is a Python library for topic modeling and document similarity analysis.",
    "Latent Semantic Analysis (LSA) is a technique used for extracting hidden semantic structures in a document collection.",
    "The Vector Space Model (VSM) is fundamental for tasks like document similarity and topic modeling.",
]

tokenized_corpus=[doc.lower().split() for doc in documents]
dictionary=corpora.Dictionary(tokenized_corpus)
corpus=[dictionary.doc2bow(doc) for doc in tokenized_corpus]
tfidf_model = TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]
num_topics = 2
lsa_model = LsiModel(tfidf_corpus, num_topics=num_topics)
pprint(lsa_model.print_topics())
term_document_matrix = lsa_model[tfidf_corpus]
for doc, as_text in zip(term_document_matrix, tokenized_docs):
    print(as_text)
    print(doc)
    print()
idf = tfidf_model.idfs
print("Inverse Document Frequency (IDF):")
for term, value in zip(dictionary.token2id.keys(), idf):
    print(f"{term}: {value}")
