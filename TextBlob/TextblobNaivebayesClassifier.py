from textblob.classifiers import NaiveBayesClassifier

train_data = [
    ("I love the new features of this app!", 'positive'),
    ("The customer support is excellent.", 'positive'),
    ("This product is a disappointment.", 'negative'),
    ("I can't recommend this service to anyone.", 'negative'),
]

classifer=NaiveBayesClassifier(train_data)
test_data="the product is not bad"
classification=classifier.classify(test_data)
print(classification)
