import spacy
text_cat=spacy.load("en_core_web_sm")
text=""
doc=text_cat(text)
print(doc.cats) 
