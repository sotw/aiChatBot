import sqlite3
import numpy as np
import pickle
import jieba
from fugashi import Tagger
from gensim.models import KeyedVectors
from gensim.models import fasttext
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    
    return " ".join(refined_words)

# --- 1. SETTINGS & LOADING ---
W2V_PATH = 'cjk_english_300.bin'
DB_NAME = 'chatbot_data.db'

print("Loading Model (this may take a minute)...")
# limit=200000 speeds up loading while keeping the most common words
# w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True, limit=200000)
#ft_kv = fasttext.load_facebook_vectors(W2V_PATH)
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

# Save the label map for the chatbot to use later
with open('label_map.pkl', 'wb') as f:
    pickle.dump(unique_labels, f)

# --- 3. Tokenizer
# 3-1. Fit the tokenizer on your text patterns

segmented_patterns = [tokenize_cjk(p) for p in patterns]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(segmented_patterns)

# 3-2. Convert text to sequences of numbers (e.g., "hello" -> 5)
sequences = tokenizer.texts_to_sequences(segmented_patterns)

# 3-3. Pad sequences so they are all exactly 20 words long
train_x = pad_sequences(sequences, maxlen=20)
train_y = to_categorical([label_map[t] for t in tags])
# 3-4. Define vocab_size (Total unique words + 1 for padding)
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    try:
        # Check if the word exists in your loaded KeyedVectors
        embedding_matrix[i] = ft_kv[word]
    except KeyError:
        # Word not found, stays as zeros
        pass



# --- 4. BUILD & TRAIN MODEL ---
model = Sequential([
    # vocab_size is now the count from your tokenizer
    Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],mask_zero=True,trainable=False),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=64, verbose=1)

model.save('chatbot_brain.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Training Complete. Model and Label Map saved!")


with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Training Complete. Model and Label Map saved!")
