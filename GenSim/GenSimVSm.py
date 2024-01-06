from gensim import corpora
from gensim.models import TfidfModel
corpus = [
    "Gensim is an open-source library for topic modeling.",
    "Vector space models are essential in natural language processing.",
    "The TfidfModel in Gensim calculates the term-document matrix.",
    "Inverse document frequency (IDF) is crucial for information retrieval.",
]


tokenized_corpus=[doc.split() for doc in corpus]
dictionary=corpora.Dictionary(tokenized_corpus)
term_document=[dictionary.doc2bow(doc) for doc in tokenized_corpus]
tfid_model=TfidfModel(term_document)
tfid_matrix=tfid_model[term_document]
print("Term-Document Matrix:")
for doc in tfidf_matrix:
    print(doc)
print("\nInverse Document Frequency (IDF):")
for term, idf_value in tfidf_model.idfs.items():
    print(f"{dictionary[term]}: {idf_value}")
