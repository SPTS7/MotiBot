#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#%%
model_name="Motibot.keras"

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


#%%

try:
    model = tf.keras.models.load_model(model_name)
except:
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
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
    adam = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(xs, ys, epochs=10, verbose=1,callbacks=[earlystop])
    model.save(model_name)
    #print model.summary()
    #print(model)




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
    seed_text = question
    keep=True
    response = ""
    while keep:
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list,verbose = 0), axis=-1)
        prediction = np.max(model.predict(token_list,verbose = 0), axis=-1)
        if prediction > 0.5:
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
    return response

#%%

a=True
# Start chatbot
while a:
    question = input('User: ')
    answer = predict_answer(model, tokenizer, question)
    print('User:',question)
    print('Bot:', answer)
    #a=False

# %%
