import sqlite3
import numpy as np
import pickle
import jieba
from fugashi import Tagger
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
import collections
from torch.utils.data import DataLoader, TensorDataset

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, unique_labels_count):
        super(TextClassifier, self).__init__()
        
        # 1. Embedding Layer
        # We load the weights and set requires_grad=False (trainable=False)
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), 
            freeze=True, 
            padding_idx=0 # Equivalent to mask_zero=True
        )
        
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


# Initialize Japanese Tagger
tagger = Tagger('-Owakati')

def tokenize_cjk(text):
    # 1. Use Fugashi for Japanese segmentation
    # (Fugashi is generally safe for mixed Kanji, but we can refine)
    words = tagger.parse(text).split()
    
    # 2. Use Jieba to further refine Chinese segments if needed
    # We join and re-split to ensure both engines' strengths are used
    refined_words = []
    for w in words:
        refined_words.extend(jieba.lcut(w))
    
    return refined_words

# --- 1. SETTINGS & LOADING ---
DB_NAME = 'chatbot_data.db'

print("Loading Model (this may take a minute)...")
# limit=200000 speeds up loading while keeping the most common words
#from gensim.models import KeyedVectors

# This is nearly INSTANT and uses very little RAM because of mmap='r'
ft_kv = KeyedVectors.load('cjk_fasttext.kv', mmap='r')
'''
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    if word in ft_kv:
        embedding_matrix[i] = fg_kv[word]
    else:
        try:
            embeddeding_matrix[i] = ft_kv.get_vector(word)
        except KeyError:
            pass
'''

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

# --- 3. Tokenizer
# 3-1. Fit the tokenizer on your text patterns

segmented_patterns = [tokenize_cjk(p) for p in patterns]

word_counts = collections.Counter()
for seq in segmented_patterns:
    word_counts.update(seq)

sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
word_index = {word: i + 1 for i, (word, count) in enumerate(sorted_words)}

sequences = [[word_index[word] for word in seq if word in word_index] for seq in segmented_patterns]

def manual_pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            # Truncate from the left, pad from the left (pre-padding)
            trunc = seq[-maxlen:]
            padded[i, -len(trunc):] = trunc
    return padded

train_x_np = manual_pad_sequences(sequences, maxlen=20)
# IMPORTANT: PyTorch CrossEntropyLoss usually expects class INDICES (0, 1, 2...)
# instead of One-Hot vectors (0, 0, 1...). So we skip to_categorical.
train_y_np = np.array([label_map[t] for t in tags])

# 3-4. Define vocab_size (Total unique words + 1 for padding)
vocab_size = len(word_index) + 1

embedding_matrix = np.zeros((vocab_size, 300))
for word, i in word_index.items():
    if word in ft_kv:
        # Check if the word exists in your loaded KeyedVectors
        embedding_matrix[i] = ft_kv[word]
#--- 3-5. THE PYTORCH BRIDGE (The New Part) ---
# Convert NumPy arrays to PyTorch Tensors
X_tensor = torch.LongTensor(train_x_np) 
Y_tensor = torch.LongTensor(train_y_np)

dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- 4. BUILD MODEL ---
model = TextClassifier(vocab_size, embedding_matrix, len(unique_labels))

# --- 5. SETUP (Replaces model.compile) ---
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss() # Combines LogSoftmax and NLLLoss

# --- 6. TRAINING LOOP (Replaces model.fit) ---
model.train()
for epoch in range(200):
    # You need to wrap your data in a DataLoader first!
    for texts, labels in train_loader: 
        optimizer.zero_grad()           # Clear old gradients
        outputs = model(texts)          # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward()                 # Backward pass (Backprop)
        optimizer.step()                # Update weights
    
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

# --- 7. SAVE ---
torch.save(model.state_dict(), 'chatbot_brain.pth')

# Save the label map for the chatbot to use later
with open('label_map.pkl', 'wb') as f:
    pickle.dump(unique_labels, f)

# Save the word index (to turn "hello" -> ID 5)
with open('tokenizer_word_index.pkl', 'wb') as f:
    pickle.dump(word_index, f)

# Print first 10 items to see the format
print("Sample keys in word_index:")
print(list(word_index.keys())[:10])
