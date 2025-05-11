import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Okay, let's load up the recipe data from our CSV.
df = pd.read_csv('./datasets/data.csv', encoding='latin1')

# Now, we're going to glue together the dish name, ingredients, and procedure.
# Basically, we want the model to see the recipe as one big chunk of text.
df['text'] = df['Dish Name'] + " " + df['Ingredients'] + " " + df['Procedure']

# Let's grab all the text we've created and get it ready for the model.
texts = df['text'].values

# Time to turn words into numbers. The tokenizer does this for us.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1  # we are adding padding to make equal length 

# We're making sequences of words, like "add the", then "add the salt", and so on.
# This helps the model learn to predict the next word.
input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]  # Turn text into numbers
    for i in range(2, len(token_list) + 1): 
        n_gram_sequence = token_list[:i]  
        input_sequences.append(n_gram_sequence)

# Gotta make sure all sequences are the same length, so we find the longest one.
max_sequence_len = max([len(seq) for seq in input_sequences])

# And we pad the shorter ones with zeros at the beginning.
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Split the sequences into inputs (X) and outputs (y).
# X is everything except the last word, and y is the last word.
X = input_sequences[:, :-1]
labels = input_sequences[:, -1]
y = to_categorical(labels, num_classes=total_words)  

# Here's where we build our model, an LSTM network.
model = Sequential()
model.add(Embedding(total_words, 100, input_length=X.shape[1]))  # Embed words into vectors
model.add(LSTM(150, return_sequences=True))  # First LSTM layer, keeps the sequences
model.add(Dropout(0.2))  # Dropout layer
model.add(LSTM(100))  # Second LSTM layer, just the final output
model.add(Dense(total_words, activation='softmax')) 


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

# We'll stop training early if the loss stops improving.
early_stop = EarlyStopping(monitor='loss', patience=3)

# Train the model, finally!
history = model.fit(X, y, epochs=20, batch_size=128, callbacks=[early_stop])

# Save the trained model and the tokenizer, so we can use them later.
model.save("./trained_model/recipe_generator_model.h5")
with open('./trained_model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Training complete. Model and tokenizer saved.")