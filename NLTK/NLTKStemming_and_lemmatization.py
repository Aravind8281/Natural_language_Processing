#NLTK provides tools for reducing words to their base or root form. This can be useful for text normalization
import nltk
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter=PorterStemmer()
lemmatizer=WordNetLemmatizer()
word="Playing"
porter_word=porter.stem(word)
lematize_word=lemmatizer.lemmatize(word)
print(porter_word)
print(lematize_word)
#play
#Playing
