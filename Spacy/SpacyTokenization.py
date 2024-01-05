import spacy
nlp=spacy.load("en_core_web_sm")
doc=nlp("this is spacy modulw learning ")
for token in doc:
  print(token.text)
