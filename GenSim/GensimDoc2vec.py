from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
documents = [
    "Doc2Vec is an extension of Word2Vec.",
    "It represents entire documents as vectors.",
    "Doc2Vec is used for document similarity and categorization."
]
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(documents)]
model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=100)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
vector = model.dv['0']
similar_documents = model.dv.most_similar('0', topn=2)
print("Vector for document 0:", vector)
print("Similar documents to document 0:", similar_documents)
