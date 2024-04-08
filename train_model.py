import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from keras.models import model_from_json


lemmatizer = WordNetLemmatizer()
# Load the intents JSON data
with open('intents.json') as file:
    data = json.load(file)

# Extract data from JSON
intents = data['intents']

# Initialize empty lists for training data
training_x = []
training_y = []

# Initialize empty lists for classes and words
words = []
classes = []
documents = []
ignoreCharacters = [ ',', '.', '!', '?' ]

# for intent in intents:
#     for pattern in intent ['patterns']:
#         wordList = nltk.word_tokenize (pattern)
#         words.extend (wordList)
#         documents.append ( (wordList, intent ['tag']) )
#         if intent ['tag'] not in classes:
#             classes.append (intent ['tag'])

# words = [ lemmatizer.lemmatize (word) for word in words if word not in ignoreCharacters]
# words = sorted (set (words))

# classes = sorted ( set (classes))

# pickle.dump (words, open ('words.pkl', 'wb'))
# pickle.dump (classes, open ('classes.pkl', 'wb'))

# classes = []
# words = []

# Loop through intents
for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = pattern.lower().split()
        words.extend(word_list)
        # Add to training data
        training_x.append(word_list)
        training_y.append(intent['tag'])
    # Add intent tag to classes list
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Unique words and classes
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

words = [ lemmatizer.lemmatize (word) for word in words if word not in ignoreCharacters]
pickle.dump (words, open ('words.pkl', 'wb'))
pickle.dump (classes, open ('classes.pkl', 'wb'))
# Create training data
x_train = []
y_train = []

for idx, doc in enumerate(training_x):
    # Initialize bag of words for each pattern
    bag = []
    # Convert pattern to one-hot encoded bag of words
    for word in words:
        bag.append(1) if word in doc else bag.append(0)
    # Convert intents to one-hot encoded labels
    output_row = [0] * len(classes)
    output_row[classes.index(training_y[idx])] = 1
    x_train.append(bag)
    y_train.append(output_row)

# Convert to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Define the model
model = Sequential([
    Dense(128, input_shape=(len(x_train[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(y_train[0]), activation='softmax')
])

# Compile the model
sgd = SGD (learning_rate = 0.01, decay = 1e-6, momentum = 0.5, nesterov = True)
model.compile (loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs = 200, batch_size = 5, verbose = 1)

# Save the model
model.save('mental_health_chatbot_model.h5')

print("Model trained and saved successfully!")
