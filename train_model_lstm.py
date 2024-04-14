import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(42)

# Load intents JSON data
with open('intents.json') as file:
    data = json.load(file)
print 
# Extract data from JSON
intents = data['intents']

# Initialize empty lists for training data
training_x = []
training_y = []

# Initialize empty lists for classes and words
words = []
classes = []
documents = []
ignoreCharacters = [',', '.', '!', '?']

lemmatizer = WordNetLemmatizer()

# Loop through intents
for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to training data
        training_x.append(word_list)
        training_y.append(intent['tag'])
    # Add intent tag to classes list
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize words and remove ignore characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreCharacters]
# Remove duplicates and sort
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

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
    LSTM(32, return_sequences=True, input_shape=(len(x_train[0]), 1)),
    BatchNormalization(),
    LSTM(32, return_sequences=True),
    BatchNormalization(),
    LSTM(32),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(x_train, np.argmax(y_train, axis=1), epochs=1500, batch_size=50, verbose=1, validation_split=0.2, 
                    #callbacks=[early_stopping]
                    )

# Save the model
model.save('mental_health_chatbot_model.keras')

print("Model trained and saved successfully!")
