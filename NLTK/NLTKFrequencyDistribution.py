# Frequency Distribution helps to count number of words in a sentence
from nltk import FreqDist
sentence="Aravind is a emerging Artificial Intelligence Developer"
words=word_tokenize(sentence)
freq=FreqDist(words)
print(freq.most_common(5))
#[('Aravind', 1), ('is', 1), ('a', 1), ('emerging', 1), ('Artificial', 1)]
