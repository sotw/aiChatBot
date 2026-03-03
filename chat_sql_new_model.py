import numpy as np
import pickle
import sqlite3
import random
import MeCab
import jieba
import re
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

G_SPEAK_LANG = 'en'
# --- 1. CONFIGURATION ---
W2V_PATH = 'cjk_english_300.bin'
MODEL_PATH = 'chatbot_brain.h5'
LABEL_MAP_PATH = 'label_map.pkl'
DB_NAME = 'chatbot_data.db'

translator_ja = GoogleTranslator(source='en', target='ja')
translator_zh = GoogleTranslator(source='en', target='zh-TW')

print("Initializing Chatbot... (Loading 300-dim Word2Vec)")
# Loading the Google News model
w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)

print("Loading Neural Network and Labels...")
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, 'rb') as f:
    unique_labels = pickle.load(f)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize MeCab once (Global)
# -Owakati tells MeCab to return only the separated words
tagger = MeCab.Tagger("-Owakati")

# --- 2. THE VECTORIZER (300-dim) ---
def get_sentence_vector(text):
    # Detect CJK characters
    has_chinese = re.search(r'[\u4e00-\u9fff]', text)
    has_japanese = re.search(r'[\u3040-\u30ff]', text) # Hiragana/Katakana

    if has_japanese:
        # Use MeCab for Japanese
        words = tagger.parse(text).strip().split()
    elif has_chinese:
        # Use Jieba for Chinese
        words = list(jieba.cut(text))
    else:
        # Standard English splitting
        words = text.lower().split()
    
    # 300-dim vector aggregation
    vectors = [w2v[w] for w in words if w in w2v]
    
    if not vectors:
        # Fallback: Character-level for CJK if word-level fails
        if has_chinese or has_japanese:
            vectors = [w2v[char] for char in text if char in w2v]
            
        if not vectors:
            return np.zeros(300)
        
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
    elif user_input.lower() == 'speak english':
       G_SPEAK_LANG = 'en'
    elif user_input.lower() == 'speak chinese':
       G_SPEAK_LANG = 'zh'
    elif user_input.lower() == 'speak japanese':
       G_SPEAK_LANG = 'ja'
# --- 1. PREPARE TEXT INPUT ---
# Convert text to integers based on the tokenizer used during training
    seq = tokenizer.texts_to_sequences([user_input])
# Ensure it is exactly the same length (20) as your model's input_length
    padded_seq = pad_sequences(seq, maxlen=20) 

# --- 2. PREDICT ---
# We pass a list [text_input, user_input] to match the Functional API
    prediction = model.predict(np.array(padded_seq), verbose=0)

# --- 4. EXTRACT RESULTS ---
    results_index = np.argmax(prediction)
    tag = unique_labels[results_index]

    confidence = prediction[0][results_index]    # Output Logic

    if confidence > 0.5:
        response = get_sql_response(tag)
        if G_SPEAK_LANG == 'zh':
            response = translator_zh.translate(response)
        elif G_SPEAK_LANG == 'ja':
            response = translator_ja.translate(response)

        print(f"Bot: {response}")
    else:
        print("Bot: I'm not quite sure. Could you try asking in a different way?")
