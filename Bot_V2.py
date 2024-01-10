import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Download NLTK data
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

tokenizer = Tokenizer()

# Load data
with open('data.txt', 'r', encoding='utf-8') as f:
    data = f.read()


corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

#print(tokenizer.word_index)
#print(total_words)


input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 300, input_length=max_sequence_len-1),
    tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.Conv1D(64, 5, activation='relu'),
    #tf.keras.layers.MaxPooling1D(pool_size=5),
    tf.keras.layers.LSTM(150),
    #tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(total_words, activation='softmax')
])


adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=10, verbose=1)
#print model.summary()
#print(model)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')

#question = "I need motivation to exercise"

# Define function to predict answer
def predict_answer(model, tokenizer, question):
    seed_text = question
    next_words = 10
    response = ""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        response += " " + output_word
    return response



a=True
# Start chatbot
while a:
    question = input('User: ')
    answer = predict_answer(model, tokenizer, question)
    print('User:',question)
    print('Bot:', answer)
    a=False
