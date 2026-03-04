import numpy as np
import pickle
import sqlite3
import random
import MeCab
import jieba
import re
import torch
from gensim.models import KeyedVectors
from deep_translator import GoogleTranslator

import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, unique_labels_count):
        super(TextClassifier, self).__init__()
        
        # 1. Embedding Layer
        # We load the weights and set requires_grad=False (trainable=False
        # Check if we are providing a matrix (Training) or just a size (Inference)
        if embedding_matrix is not None:
            # --- TRAINING MODE ---
            # Convert the numpy matrix to a tensor HERE
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix), 
                freeze=True,
                padding_idx=0
            )
        else:
            # --- INFERENCE/USER SIDE MODE ---
            # Just create an empty embedding layer with the right dimensions.
            # load_state_dict() will fill these weights in later.
            self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)

        # 2. Spatial Dropout
        # PyTorch's Dropout2d on (N, C, L) data acts like SpatialDropout1d
        self.spatial_dropout = nn.Dropout2d(0.2)
        
        # 3. LSTM
        # Note: PyTorch LSTM doesn't support 'recurrent_dropout' natively.
        # This standard dropout applies to the output of each layer.
        self.lstm = nn.LSTM(
            input_size=300, 
            hidden_size=64, 
            batch_first=True, 
            dropout=0.2 if 1 > 1 else 0 # Dropout only works if layers > 1
        )
        
        # 4. Dense (Linear) Layer
        self.fc = nn.Linear(64, unique_labels_count)
        self.softmax = nn.LogSoftmax(dim=1) # Recommended for use with NLLLoss

    def forward(self, x):
        # Embedding output: [Batch, Seq_Len, 300]
        x = self.embedding(x)
        
        # Spatial Dropout needs [Batch, Channels, Height, Width] or [Batch, Features, Seq_Len]
        # So we permute, drop, and permute back
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        x = self.spatial_dropout(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        
        # LSTM returns (output, (hidden_state, cell_state))
        # We take the last hidden state for classification
        _, (h_n, _) = self.lstm(x)
        
        # Final Dense layer
        out = self.fc(h_n[-1])
        return self.softmax(out)


global G_SPEAK_LANG
G_SPEAK_LANG = 'en'
# --- 1. CONFIGURATION ---
MODEL_PATH = 'chatbot_brain.h5'
LABEL_MAP_PATH = 'label_map.pkl'
DB_NAME = 'chatbot_data.db'

translator_ja = GoogleTranslator(source='en', target='ja')
translator_zh = GoogleTranslator(source='en', target='zh-TW')

print("Initializing Chatbot... (Loading 300-dim Word2Vec)")

print("Loading Neural Network and Labels...")
# 1. Load your metadata first
with open(LABEL_MAP_PATH, 'rb') as f:
    unique_labels = pickle.load(f)  # Gives us the count for the last layer

with open('tokenizer_word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)    # Gives us the vocab_size for the first layer

# 2. Parameters for the "Brain"
v_size = len(word_index) + 1
out_size = len(unique_labels)
emb_dim = 300

# 3. Initialize model with placeholders (zeros)
# We don't need the actual ft_kv matrix here because the .pth file has the weights!
model = TextClassifier(vocab_size=v_size, embedding_matrix=None, unique_labels_count=out_size)

# 4. Load the trained weights
model.load_state_dict(torch.load('chatbot_brain.pth'))
print("Debug - First 5 weights of the output layer:")
print(model.fc.weight[0][:5])
model.eval()

# Initialize MeCab once (Global)
# -Owakati tells MeCab to return only the separated words
tagger = MeCab.Tagger("-Owakati")

# --- 2. THE VECTORIZER (300-dim) ---
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

def prepare_input(user_text, word_index, max_len=20):
    # 1. CJK Logic (Fugashi + Jieba)
    # Matches your training preprocessing exactly
    words = tagger.parse(user_text).split()
    refined_words = []
    for w in words:
        refined_words.extend(jieba.lcut(w))
# --- DEBUG START ---
    print(f"Tokens found: {refined_words}")
    sequence = []
    for w in refined_words:
        clean_w = w.lower().strip()
        idx = word_index.get(clean_w, 0)
        print(f"  '{clean_w}' -> ID: {idx}")
        sequence.append(idx)
    # --- DEBUG END ---    
    # 2. Transform to integers using word_index (Safe lookup)
    # Using .get(word, 0) handles "Out Of Vocabulary" (OOV) words
    sequence = [word_index.get(w.lower().strip(), 0) for w in refined_words]
    
    # 3. Pad (Manual implementation of 'pre' padding)
    # Start with an array of zeros
    padded = np.zeros(max_len, dtype=int)
    if len(sequence) > 0:
        # Take the last 'max_len' tokens (truncating from the front)
        trunc = sequence[-max_len:]
        # Place them at the end of the zero array (padding at the front)
        padded[-len(trunc):] = trunc
    
    # 4. Convert to PyTorch Tensor and add batch dimension [1, max_len]
    return torch.LongTensor(padded).unsqueeze(0)

# --- 4. THE CHAT LOOP ---
print("\n--- BOT IS ONLINE! (Type 'quit' to stop) ---")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

# --- 1. PREPARE TEXT INPUT ---
    processed_input = prepare_input(user_input, word_index)

# --- 2. PREDICT ---
    with torch.no_grad(): # Use no_grad to save memory and speed up
        # We call the model directly
        output = model(processed_input)
        print(f"Raw Output (Logits): {output}") # See if all values are the same
        # Convert LogSoftmax back to 0-1 probability
        probs = torch.exp(output)
        print(f"Probabilities: {probs}")
        # Get the highest probability (confidence) and its index
        conf_tensor, idx_tensor = torch.max(probs, dim=1)
        
        # Convert Tensors to standard Python numbers
        confidence = conf_tensor.item()
        results_index = idx_tensor.item()

    # --- 3. EXTRACT RESULTS ---
    tag = unique_labels[results_index]

    if confidence > 0.6:
        response = get_sql_response(tag)
        action = get_sql_action(tag)
        paras = get_sql_para(tag)
        if action!=None and paras != None:
            bot_action(action, paras)
        '''
        if G_SPEAK_LANG == 'zh':
            response = translator_zh.translate(response)
        elif G_SPEAK_LANG == 'ja':
            response = translator_ja.translate(response)
        '''

        print(f"Bot: {response}")
    else:
        response = f"Bot: I'm not quite sure about:{user_input}. Could you try asking in a different way?"
        '''
        if G_SPEAK_LANG == 'zh':
            response = translator_zh.translate(response)
        elif G_SPEAK_LANG == 'ja':
            response = translator_ja.translate(response)
        '''
        print(response)
