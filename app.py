import streamlit as st
import pickle

import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    #for removing special characters
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("Email/SMS Spam Classifier")

sms = st.text_area("enter the message")

if st.button("Predict"):

    #1. preprocess
    transform_sms = transform_text(sms)

    #2. vectorize
    vector_input = tfidf.transform([transform_sms])

    #3. predict
    result = model.predict(vector_input)[0]

    #4. display
    if (result == 1):
        st.header("Spam")
    else:
        st.header("Not spam")