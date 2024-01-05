#NLTK provides tools for breaking down text into individual words or sentences
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
nltk.download('punkt')
text=" hello people im aravindan machine learning developer"
word=word_tokenize(text)
sentence=sent_tokenize(text)
print(word)
print(sentence)

#['hello', 'people', 'im', 'aravindan', 'machine', 'learning', 'developer']
#[' hello people im aravindan machine learning developer']
