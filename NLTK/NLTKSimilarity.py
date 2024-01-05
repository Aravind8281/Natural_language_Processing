# measure similarity between words and provide output
from nltk.corpus import wordnet
from nltk.wsd import lesk
word1=wordnet.synsets("ship")[0]
word2=wordnet.synsets("yacht")[0]
similarity=word1.wup_similarity(word2)
print(similarity)
#0.90
