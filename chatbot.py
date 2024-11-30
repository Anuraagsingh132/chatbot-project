import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents, words, and classes from files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load or train the model
try:
    model = load_model('chatbot_model.h5')
except:
    # Train the model if it doesn't exist
    training = []
    output_empty = [0] * len(classes)

    # Prepare training data
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            sentence_words = nltk.word_tokenize(pattern)
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            bag = [0] * len(words)
            for s in sentence_words:
                for i, w in enumerate(words):
                    if w == s:
                        bag[i] = 1
            output_row = list(output_empty)
            output_row[classes.index(intent['tag'])] = 1
            training.append([bag, output_row])

    # Shuffle training data and prepare input/output sets
    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])

    # Compile the model
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    # Save the trained model
    model.save('chatbot_model.h5')


# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Convert sentence into bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predict the class of the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


# Get the response based on the predicted class
def get_response(intents_list, intents_json):
    if not intents_list:  # Check if intents_list is empty
        return "Sorry, I didn't understand that. Could you ask differently?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Main function to generate chatbot response
def chatbot_response(msg):
    ints = predict_class(msg)
    res = get_response(ints, intents)
    return res


# Test the chatbot
if __name__ == "__main__":
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            break
        response = chatbot_response(message)
        print("Bot:", response)
