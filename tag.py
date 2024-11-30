import json

# Load the intents file
with open('intents.json') as file:
    intents = json.load(file)

# Extract tags and print each on a new line
for intent in intents['intents']:
    print(intent['tag'])
