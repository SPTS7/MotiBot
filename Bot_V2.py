#%%
import numpy as np
import tensorflow as tf
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


#%%
model_name="Motibot2.keras"

import os
data = ""
def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


for file in get_files('Training'):
    # Load data
    with open('Training/'+file, 'r', encoding='utf-8') as f:
        data += f.read()
        data += "\n"


#%%
        
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
processed_data = [preprocess(qa) for qa in data.split('\n')]

#%%
#corpus = data.lower().split("\n")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_data)
total_words = len(tokenizer.word_index) + 1

#print(tokenizer.word_index)
#print(total_words)


input_sequences = []
for line in processed_data:
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


#%%

try:
    model = tf.keras.models.load_model(model_name)
except:
    # Build model
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 5000, input_length=max_sequence_len-1),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(50, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=5),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(total_words, activation='softmax')
    ])
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
    adam = Adam(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(xs, ys, epochs=1, verbose=1,callbacks=[earlystop])
    model.save(model_name)
    #print model.summary()
    #print(model)``




def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

#plot_graphs(history, 'accuracy')

#question = "I need motivation to exercise"

#%%

# Define function to predict answer
def predict_answer(model, tokenizer, question):
    #seed_text = question  
    seed_text = [preprocess(qa) for qa in question.split('\n')]
    print(seed_text)
    num_words=0
    response = ""
    while num_words<10:
        token_list = tokenizer.texts_to_sequences(seed_text)[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list,verbose = 0), axis=-1)
        prediction = np.max(model.predict(token_list,verbose = 0), axis=-1)
        if prediction > 0.3:
            keep=True
        else:
             keep=False
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        response += " " + output_word
        num_words +=1 
    return response

#%%

a=True
# Start chatbot
while a:
    question = input('User: ')
    answer = predict_answer(model, tokenizer, question)
    print('User:',question)
    print('Bot:', answer)
    a=False

# %%
