#################################################################
# Input from user is taken. It is cleaned using NLP,            #
# and then it is send in the trained Neural Network model.      #
# Model predicts the tag of that sentence and then a random     #
# response corresponding to that tag is returned to user        #
#################################################################
import random
import json
import pickle
import numpy as np
from telegram.ext import CommandHandler, filters, MessageHandler, ApplicationBuilder
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads (open('intents.json').read())

words = pickle.load (open('words.pkl', 'rb'))
classes = pickle.load (open('classes.pkl', 'rb'))

model = load_model ('mental_health_chatbot_model.keras')

def cleanUpSentence (sentence):
    sentenceWords = nltk.word_tokenize (sentence)
    sentenceWords = [lemmatizer.lemmatize(word) for word in sentenceWords]
    return  sentenceWords


def bagOfWords (sentence):
    sentenceWords = cleanUpSentence (sentence)
    bag = [0] * len (words)
    for w in sentenceWords:
        for i, word in enumerate(words):
            if w == word :
                bag [i] = 1
    return np.array (bag)


def predictClass (sentence):
    bow = bagOfWords (sentence)
    res = model.predict (np.array([bow]))[0]     # 0.2 0.3 0.4 0 1 0000

    ERROR_TRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_TRESHOLD]

    results.sort (key = lambda x : x[1], reverse = True)
    returnList = []
    for r in results:
        returnList.append ({'intent' : classes[r[0]], 'probablity' : str(r[1])})
    return returnList


def getResponse (intentsList, intentsJson):
    tag = intentsList[0]['intent']
    print(tag)
    listOfIntents = intentsJson['intents']
    #result = "I'm sorry, I didn't understand that."
    for i in listOfIntents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

print ('Bot is Running !!')

# while (True):
#     message = input ("You : ")
#     ints = predictClass (message)
#     res = getResponse (ints, intents)
#     print ('TMI : ', res)
# Function to handle the /start command
async def start(update, context):
    await update.message.reply_text("Hello! I'm your mental health bot. How can I help you today?")
# Function to handle incoming messages
async def handle_message(update, context):
    user_input = update.message.text
    print(user_input)
    intent = predictClass(user_input)
    response = getResponse (intent, intents)
    await update.message.reply_text(response)

def main():
    # Initialize the Updater and pass in your bot's token
    app = ApplicationBuilder().token("7123912517:AAG2btyXGdy2x6x3hM79vss5KKUS_B4b8Ww").build()

    # Register a command handler for the /start command
    app.add_handler(CommandHandler("start", start))

    # Register a message handler to handle incoming messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    # Start the Bot
    app.run_polling()

    # Run the bot until you press Ctrl-C
    app.idle()

if __name__ == "__main__":
    main()