**Spam Message Classifier -**
A Machine Learning–based spam classifier that detects whether a message is Spam or Not Spam. The model is deployed as an interactive Streamlit web app and hosted on Render.

**Live Demo - App Link:** https://sms-email-spam-classifier-3-pzws.onrender.com/

**Project Overview -**
This project uses Natural Language Processing (NLP) techniques and a supervised ML algorithm to classify text messages as spam or ham.
The trained model is integrated into a Streamlit application that allows users to test messages in real time.

**Tech Stack -**
Python
Scikit-learn
Pandas 
NumPy
NLTK
Streamlit
Render (Deployment)

**Machine Learning Approach -**
Text preprocessing (lowercasing, tokenization, stopword removal, stemming)
Feature extraction using TF-IDF Vectorizer
Model training using Naive Bayes (MultinomialNB)
Model serialization using pickle
