import joblib


# Load trained model
model = joblib.load("chatbot_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Chatbot responses
responses = {
    "greeting": "Hello! How can I help you?",
    "name": "I am an ML chatbot built using Python and scikit-learn.",
    "status": "I'm doing great! Thanks for asking.",
    "capability": "I can answer simple questions based on my training data.",
    "goodbye": "Goodbye! Have a nice day.",
    "thanks": "You're welcome!"
}


print("================================")
print("       ML CHATBOT")
print("================================")
print("Type 'exit' to stop the chatbot.\n")


# Start conversation
while True:

    user_input = input("You: ")

    # Stop chatbot
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    # Convert user text into numbers
    user_vector = vectorizer.transform([user_input])

    # Predict intent
    intent = model.predict(user_vector)[0]

    # Get response
    response = responses.get(
        intent,
        "Sorry, I don't understand that."
    )

    print("Bot:", response)