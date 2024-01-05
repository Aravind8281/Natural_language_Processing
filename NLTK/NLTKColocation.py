#Collocation refers to the occurrence of words together more often than would be expected by chance. 
#NLTK provides a Collocations module for finding collocations in a given text
import nltk
nltk.download('genesis')
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
words=nltk.corpus.genesis.words('english-kjv.txt')
bigram=BigramCollocationFinder.from_words(words)
bigram_collocation=bigram.nbest(BigramAssocMeasures.pmi,10)
print("Top 10 Collocations :")
for col in bigram_collocation:
  print(''.join(col))
AshterothKarnaim
HittiWhich
Philistim,)
Thirtymilch
Whososheddeth
bakeunleavened
burninglamp
figleaves
fordJabbok
horseheels
