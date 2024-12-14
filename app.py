# Save this code as app.py
import streamlit as st
import pickle

# Load the trained Logistic Regression model and TF-IDF vectorizer
with open('log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Streamlit app title
st.title("Spam Mail Classifier")

# Text input field for user to enter email content
email_text = st.text_area("Enter Email Text:", height=200)

# Button to trigger the classification
if st.button("Classify"):
    if email_text:
        # Transform the email text using the TF-IDF vectorizer
        email_tfidf = tfidf_vectorizer.transform([email_text])
        
        # Make prediction using the trained model
        prediction = log_reg_model.predict(email_tfidf)[0]
        label = 'Spam' if prediction == 1 else 'Ham'
        
        # Display the prediction result
        st.subheader(f"Prediction: {label}")
    else:
        st.warning("Please enter some text to classify.")

