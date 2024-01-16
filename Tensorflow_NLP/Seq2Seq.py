import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# Define the Seq2Seq model
latent_dim = 256

encoder_inputs = Input(shape=(None,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(embedding_layer)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embedding_layer, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

