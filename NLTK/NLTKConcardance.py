#Concordance is the occurrence of a word within a specific context
import nltk
from nltk.corpus import brown
from nltk.text import Text
import nltk
nltk.download('brown')
category = 'news'
words = brown.words(categories=category)
text = Text(words)
word_to_find = 'government'
text.concordance(word_to_find, lines=10)
#r per student . In addition , the government would pay a $1,000 `` cost of edu
#in the 1961-62 budget for direct government research in medicine . The Presid
#e outcome in Laos was a coalition government susceptible of Communist dominati
#eutralized Laos . The pro-Western government , which the United States had hel
# Souvanna Phouma `` neutralist '' government , never did appear to spark much 
#es would not back the pro-Western government to the hilt . If the administrati
#m , Mr. Hawksley said the federal government would pay half the salary of a fu
#eligible to apply to the federal government for financial aid in purchasing e
# the local level from the federal government . Rhode Island is going to examin
# `` great revenue '' to the local government . The council advised the governo
