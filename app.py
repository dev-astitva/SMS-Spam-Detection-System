import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    r = []
    for i in text:
        if i.isalnum():
            r += [i]

    text = r[:]
    r.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            r.append(i)

    text = r[:]
    r.clear()
    for i in text:
        r.append(ps.stem(i))

    return " ".join(r)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Spam Classifier")

inp=st.text_area("Enter your message")

if st.button("Predict"):
    #preprocess
    transformed_msg=transform_text(inp)
    #vectorize
    vector_inp=tfidf.transform([transformed_msg])
    #predict
    result=model.predict(vector_inp)[0]
    #display
    if result==0:
        st.header("Not Spam")
    else:
        st.header("Spam")
