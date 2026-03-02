import numpy as np
import pickle
import sqlite3
import random
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model

# --- 1. CONFIGURATION ---
W2V_PATH = 'word2vec-google-news-300.gz'
MODEL_PATH = 'chatbot_brain.h5'
LABEL_MAP_PATH = 'label_map.pkl'
DB_NAME = 'chatbot_data.db'

print("Initializing Chatbot... (Loading 300-dim Word2Vec)")
# Loading the Google News model
w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True, limit=200000)

print("Loading Neural Network and Labels...")
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, 'rb') as f:
    unique_labels = pickle.load(f)

# --- 2. THE VECTORIZER (300-dim) ---
def get_sentence_vector(text):
    words = text.lower().split()
    vectors = [w2v[w] for w in words if w in w2v]
    
    if not vectors:
        return np.zeros(300) # Must be 300 to avoid the (1, 90) error
        
    return np.mean(vectors, axis=0)

# --- 3. THE SQL RESPONSE FETCH ---
def get_sql_response(tag):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Fetch responses for the predicted tag
    cursor.execute("SELECT responses FROM intents WHERE tag=?", (tag,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        # Split the pipe-separated responses and pick one at random
        responses_list = row[0].split('|')
        return random.choice(responses_list)
    return "I found the intent, but no response was defined in the database."

# --- 4. THE CHAT LOOP ---
print("\n--- BOT IS ONLINE! (Type 'quit' to stop) ---")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
        
    # Process text
    vec = get_sentence_vector(user_input)
    prediction_input = np.array([vec]) # Shape (1, 300)
    
    # Predict
    prediction = model.predict(prediction_input, verbose=0)
    results_index = np.argmax(prediction)
    tag = unique_labels[results_index]
    confidence = prediction[0][results_index]

    # Output Logic
    if confidence > 0.85:
        response = get_sql_response(tag)
        print(f"Bot: {response}")
    else:
        print("Bot: I'm not quite sure. Could you try asking in a different way?")
