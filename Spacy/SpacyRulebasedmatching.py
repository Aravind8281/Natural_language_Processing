from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
matcher.add("HelloWorld", [pattern])
matches = matcher(doc)
print(matches)
