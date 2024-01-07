import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
loaded_cbow_model = Word2Vec.load("CBOW_model")
missing_word_sentence = f"I love __ language processing with {chosen_word}."
tokenized_missing_word_sentence = word_tokenize(missing_word_sentence.lower())
missing_word_index = tokenized_missing_word_sentence.index("__")
context_words = tokenized_missing_word_sentence[
    max(0, missing_word_index - 2) : missing_word_index
] + tokenized_missing_word_sentence[missing_word_index + 1 :]
predicted_word = loaded_cbow_model.predict_output_word(context_words)
print(f"Predicted sentence: I love {predicted_word[0]} language processing with {chosen_word}.")
