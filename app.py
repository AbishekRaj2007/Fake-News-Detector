import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open(r"C:\Users\ABISHEK RAJ\Desktop\MachineLearning\Learn\model.pkl", "rb"))
vectorizer = pickle.load(open(r"C:\Users\ABISHEK RAJ\Desktop\MachineLearning\Learn\vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")
st.write("Enter a news article below and check if it's fake or real.")

# Input text area
user_input = st.text_area("Paste your news article here:")

# When button is clicked
if st.button("Check Now"):
    if user_input.strip() == "":
        st.warning("Please enter some news content first.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.success("🟢 This is REAL news.")
        else:
            st.error("🔴 This is FAKE news.")
            
