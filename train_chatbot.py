import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# --- 1. DATA PREPROCESSING ---
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Load the intents file
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add to documents
        documents.append((w, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words")

# Save words and classes for the chat script to use later
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# --- 2. CREATE TRAINING DATA ---
train_x = []
train_y = []
# Create an empty array for our output
output_empty = [0] * len(classes)

for doc in documents:
    # Bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (one-hot encoding)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    train_x.append(bag)
    train_y.append(output_row)

# Convert to NumPy arrays separately (Fixes the ValueError)
train_x = np.array(train_x)
train_y = np.array(train_y)

# --- 3. DEFINE MODERN OPTIMIZER (Exponential Decay) ---
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000, 
    decay_rate=0.96,
    staircase=True
)

optimizer = SGD(
    learning_rate=lr_schedule, 
    momentum=0.9, 
    nesterov=True
)

# --- 4. BUILD THE MODEL ---
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# --- 5. TRAINING ---
# We use fit directly on train_x and train_y
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model.h5', hist)
print("Model training complete and saved as chatbot_model.h5")
