#NLTK provides tools for extracting named entities (such as names of people, organizations, and locations) from text
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import ne_chunk
sentence="Aravind is a emerging Artificial intelligence Engineer"
words=word_tokenize(sentence)
pos=pos_tag(words)
NER=ne_chunk(pos)
print(NER)
#(PERSON Aravind/NNP)
#is/VBZ
# a/DT
# emerging/VBG
# (ORGANIZATION Artificial/JJ)
# intelligence/NN
# Engineer/NNP)
