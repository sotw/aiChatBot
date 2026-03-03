import sqlite3
import numpy as np
import pickle
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. SETTINGS & LOADING ---
W2V_PATH = 'cjk_english_300.bin'
DB_NAME = 'chatbot_data.db'

print("Loading Google News Model (this may take a minute)...")
# limit=200000 speeds up loading while keeping the most common words
w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True, limit=200000)

# --- 2. FETCH DATA FROM SQL ---
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT p.pattern_text, i.tag FROM patterns p JOIN intents i ON p.intent_id = i.id")
data = cursor.fetchall()
conn.close()

patterns = [row[0] for row in data]
tags = [row[1] for row in data]

# Create a unique list of labels
unique_labels = sorted(list(set(tags)))
label_map = {label: i for i, label in enumerate(unique_labels)}

# Save the label map for the chatbot to use later
with open('label_map.pkl', 'wb') as f:
    pickle.dump(unique_labels, f)

# --- 3. Tokenizer
# 3-1. Fit the tokenizer on your text patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)

# 3-2. Convert text to sequences of numbers (e.g., "hello" -> 5)
sequences = tokenizer.texts_to_sequences(patterns)

# 3-3. Pad sequences so they are all exactly 20 words long
train_x = pad_sequences(sequences, maxlen=20)
train_y = to_categorical([label_map[t] for t in tags])
# 3-4. Define vocab_size (Total unique words + 1 for padding)
vocab_size = len(tokenizer.word_index) + 1

# --- 4. BUILD & TRAIN MODEL ---
model = Sequential([
    # vocab_size is now the count from your tokenizer
    Embedding(input_dim=vocab_size, output_dim=128, input_length=20),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5)

model.save('chatbot_brain.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Training Complete. Model and Label Map saved!")
