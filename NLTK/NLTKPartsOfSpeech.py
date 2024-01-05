#NLTK allows you to assign part-of-speech tags to words in a sentence, which is useful for understanding the grammatical structure of a text
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize
text="Aravind is a emerging Artificial intelligence Engineer"
words=word_tokenize(text)
pos=pos_tag(words)
print(pos)
#[('Aravind', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('emerging', 'VBG'), ('Artificial', 'JJ'), ('intelligence', 'NN'), ('Engineer', 'NNP')]
