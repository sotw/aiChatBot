import numpy as np
import pickle
import sqlite3
import random
import MeCab
import jieba
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

global G_SPEAK_LANG
G_SPEAK_LANG = 'en'
# --- 1. CONFIGURATION ---
MODEL_PATH = 'chatbot_brain.h5'
LABEL_MAP_PATH = 'label_map.pkl'
DB_NAME = 'chatbot_data.db'

translator_ja = GoogleTranslator(source='en', target='ja')
translator_zh = GoogleTranslator(source='en', target='zh-TW')

print("Initializing Chatbot...")

print("Loading Neural Network and Labels...")
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, 'rb') as f:
    unique_labels = pickle.load(f)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize MeCab once (Global)
# -Owakati tells MeCab to return only the separated words
tagger = MeCab.Tagger("-Owakati")

def bot_action(action, parameters):
    global G_SPEAK_LANG
    if action == None:
        return
    if action == "set":
        paras = parameters.split()
        if len(paras) == 0:
            return
        if paras[0] == 'language':
            if paras[1] == 'english':
                G_SPEAK_LANG = 'en'
            elif paras[1] == 'chinese':
                G_SPEAK_LANG = 'zh'
            elif paras[1] == 'japanese':
                G_SPEAK_LANG = 'ja'

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

def get_sql_action(tag):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Fetch responses for the predicted tag
    cursor.execute("SELECT action FROM intents WHERE tag=?", (tag,))
    row = cursor.fetchone()
    conn.close()

    if len(row)==0 or row == None:
        return None
    if row:
        if row[0] != None:
            actions_list = row[0].split('|')
            return actions_list[0]
    return None

def get_sql_para(tag):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Fetch responses for the predicted tag
    cursor.execute("SELECT parameter FROM intents WHERE tag=?", (tag,))
    row = cursor.fetchone()
    conn.close()
    
    if len(row)==0 or row == None:
        return None
    if row:
        if row[0] != None:
            para_list = row[0].split('|')
            return para_list[0]
    return None

def prepare_input(user_text, tokenizer, max_len=20):
    # 1. Use your CJK logic to add spaces (matches train side line 76)
    # Note: Use the logic from your previous message (Fugashi + Jieba)
    words = tagger.parse(user_text).split()
    refined_words = []
    for w in words:
        refined_words.extend(jieba.lcut(w))
    
    # 2. Re-join into a space-separated string
    segmented_text = " ".join(refined_words).lower().strip()
    
    # 3. Transform to integers (matches train side line 79)
    # This will NO LONGER be [[]] because the words now have spaces!
    sequence = tokenizer.texts_to_sequences([segmented_text])
    
    # 4. Pad (Use 'pre' to match Keras default)
    padded = pad_sequences(sequence, maxlen=max_len, padding='pre')
    
    return padded

def prepare_input_ori(user_text, tokenizer, max_len=20):
    # 1. Convert text to lowercase (standard for most tokenizers)
    user_text = user_text.lower().strip()
    
    # 2. Transform string to list of integers
    # 
    sequence = tokenizer.texts_to_sequences([user_text])
    
    print(f"User Text: {user_text}")
    print(f"Sequence: {sequence}")
    # 3. Pad with zeros at the BEGINNING (default 'pre')
    # Since mask_zero=True, the LSTM will ignore these 0s
    padded = pad_sequences(sequence, maxlen=max_len, padding='pre')
    
    return padded

# --- 4. THE CHAT LOOP ---
print("\n--- BOT IS ONLINE! (Type 'quit' to stop) ---")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

# --- 1. PREPARE TEXT INPUT ---
# Convert text to integers based on the tokenizer used during training
#   seq = tokenizer.texts_to_sequences([user_input])
# Ensure it is exactly the same length (20) as your model's input_length
#   padded_seq = pad_sequences(seq, maxlen=20) 
    processed_input = prepare_input(user_input, tokenizer)

# --- 2. PREDICT ---
# We pass a list [text_input, user_input] to match the Functional API
#    prediction = model.predict(np.array(padded_seq), verbose=0)
    prediction = model.predict(processed_input, verbose=0)

# --- 4. EXTRACT RESULTS ---
    results_index = np.argmax(prediction)
    tag = unique_labels[results_index]

    confidence = prediction[0][results_index]    # Output Logic
    print(f"confidence:{confidence}")

    if confidence > 0.3:
        response = get_sql_response(tag)
        action = get_sql_action(tag)
        paras = get_sql_para(tag)
        if action!=None and paras != None:
            bot_action(action, paras)
        if G_SPEAK_LANG == 'zh':
            response = translator_zh.translate(response)
        elif G_SPEAK_LANG == 'ja':
            response = translator_ja.translate(response)

        print(f"Bot: {response}")
    else:
        response = f"Bot: I'm not quite sure about:{user_input}. Could you try asking in a different way?"
        if G_SPEAK_LANG == 'zh':
            response = translator_zh.translate(response)
        elif G_SPEAK_LANG == 'ja':
            response = translator_ja.translate(response)
        print(response)
