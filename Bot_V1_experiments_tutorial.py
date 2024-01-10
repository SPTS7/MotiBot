import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

# Load data
with open('data.txt', 'r', encoding='utf-8') as f:
    raw_data = f.read()


# Preprocess data
def preprocess(data):
    # Tokenize data
    tokens = nltk.word_tokenize(data)
    
    # Lowercase all words
    tokens = [word.lower() for word in tokens]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# Preprocess data
processed_data = [preprocess(qa) for qa in raw_data.split('\n')]


# Set parameters
vocab_size = 1280
embedding_dim = 100
max_length = 40
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = len(processed_data)

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(processed_data)
word_index = tokenizer.word_index


# Create sequences
sequences = tokenizer.texts_to_sequences(processed_data)
padded_sequences = np.array(pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type))

# Create training data

#training_data = padded_sequences[:training_size]
#training_labels = padded_sequences[:training_size]

training_data, labels = padded_sequences[:,:-1],padded_sequences[:,-1]

training_labels = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)






# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length-1),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=5),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

#print(model.summary())

# Compile model

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train model
num_epochs = 10
history = model.fit(training_data, training_labels, epochs=num_epochs, verbose=1)


'''
import matplotlib.pyplot as plt
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()


plot_graphs(history, 'accuracy')

# Define function to predict answer
def predict_answer(model, tokenizer, question):
    # Preprocess question
    question = preprocess(question)
    # Convert question to sequence
    sequence = tokenizer.texts_to_sequences([question])
    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_length-1, padding=padding_type, truncating=trunc_type)
    # Predict answer
    pred = model.predict(padded_sequence)[0]
    # Get index of highest probability
    idx = np.argmax(pred)
    # Get answer
    answer = tokenizer.index_word[idx]
    return answer
'''

# Define function to predict answer
def predict_answer(model, tokenizer, question):
    for _ in range(20):
        token_list = tokenizer.texts_to_sequences([question])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        output_word += " " + output_word
        print(output_word)
    return output_word


a=True
# Start chatbot
while a:
    question = input('User: ')
    answer = predict_answer(model, tokenizer, question)
    print('Bot:', answer)
    a=False

