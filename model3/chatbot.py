import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# 1. CHATBOT TRAINING DATA
# ============================================================

data = {
    "text": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",

        "what is your name",
        "who are you",
        "tell me about yourself",

        "how are you",
        "how are you doing",

        "what can you do",
        "what can you help me with",

        "bye",
        "goodbye",
        "see you later",

        "thank you",
        "thanks"
    ],

    "intent": [
        "greeting",
        "greeting",
        "greeting",
        "greeting",
        "greeting",

        "name",
        "name",
        "name",

        "status",
        "status",

        "capability",
        "capability",

        "goodbye",
        "goodbye",
        "goodbye",

        "thanks",
        "thanks"
    ]
}


# ============================================================
# 2. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)

print("Dataset:")
print(df)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. CONVERT TEXT INTO NUMBERS
# ============================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])
y = df["intent"]

print("\nText converted into numbers")
print("Feature shape:", X.shape)


# ============================================================
# 4. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 5. CREATE ML MODEL
# ============================================================

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nChatbot model training completed")


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Chatbot Accuracy:", accuracy)


# ============================================================
# 7. SAVE MODEL
# ============================================================

joblib.dump(model, "chatbot_model.pkl")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nChatbot model saved successfully")