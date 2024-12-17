import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL and download NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Define intents
intents = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
     "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]},
    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye", "See you later", "Take care"]},
    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're welcome", "No problem", "Glad I could help"]},
    {"tag": "about", "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
     "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]},
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare training data
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define the chatbot response function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Initialize Streamlit app
st.title("AI Chatbot")
st.write("Welcome! Start chatting below.")

# Persistent session-based chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat input box
user_input = st.text_input("You:", key="user_input")

if user_input:
    bot_response = chatbot_response(user_input)
    
    # Store the conversation in session state
    st.session_state["messages"].append((f"You: {user_input}", f"Bot: {bot_response}"))
    
    # Display chat history
    for user_msg, bot_msg in st.session_state["messages"]:
        st.text_area("Chat History", value=f"{user_msg}\n{bot_msg}", height=100, max_chars=None)

    # End chat on goodbye
    if bot_response.lower() in ["goodbye", "bye"]:
        st.write("Thank you for chatting with me. Have a great day!")
        st.stop()
