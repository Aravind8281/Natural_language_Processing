import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
token=tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_sequences(corpus)
total_words=len(tokenizer.word_index)+1
sequences=tokenizer.texts_to_sequences(corpus)
input_sequences=[]
for sequence in sequences:
  n_gram_sequence=sequence[:i+1]
  input_sequences.append(n_gram_sequence)

max_sequence_length=max(len(seq) for seq in input_sequences)
padded_sequences=tf.keras.preprocessing.sequence.pad_sequences(input_sequences,maxlen=max_sequence_length,padding='pre')
X=padded_sequences[:,:-1]
y=tf.keras.utils.to_categorical(padded_sequences[:,-1],num_classes=total_words)
model=Sequential()
model.add(Embedding(input_dim=total_words,output_dim=50,input_length=max_sequence_length-1))
model.add(Flatten())
model.add(Dense(total_words,activation="softmax"))
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X,y,epochs=50,verbose=2)
ord_embeddings = model.layers[0].get_weights()[0]
word_index = tokenizer.word_index
for word, index in word_index.items():
    if index < 10:
        print(f"{word}: {word_embeddings[index]}")
