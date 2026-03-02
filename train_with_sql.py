import sqlite3
import numpy as np
import pickle
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# --- 1. SETTINGS & LOADING ---
W2V_PATH = 'word2vec-google-news-300.gz'
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

# --- 3. VECTORIZATION ---
def get_vector(text):
    words = text.lower().split()
    vectors = [w2v[w] for w in words if w in w2v]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

train_x = np.array([get_vector(p) for p in patterns])
train_y = to_categorical([label_map[t] for t in tags])

# --- 4. BUILD & TRAIN MODEL ---
model = Sequential([
    Dense(128, input_shape=(300,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('chatbot_brain.h5')
print("Training Complete. Model and Label Map saved!")
