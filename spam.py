import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def load_data():
    data = pd.read_csv("email_classification.csv") 
    return data


def train_model(data):
    X = data['email']  # can also be done using iloc
    y = data['label'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

data = load_data()
model, vectorizer, accuracy = train_model(data)

st.title("Spam Detector")
st.write(f"Model Accuracy: {accuracy:.2f}")

email_input = st.text_area("Enter the email content here:")

if st.button("Submit"):
    if email_input.strip() == "":
        st.warning("Please enter an email.")
    else:
        input_features = vectorizer.transform([email_input])

        prediction = model.predict(input_features)

        if prediction[0] == 'ham':  
            st.success("This email is classified as: Ham")
        else:
            st.error("This email is classified as: Spam")

