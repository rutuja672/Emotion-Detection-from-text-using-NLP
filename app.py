import streamlit as st
import joblib

# Page Configuration
st.set_page_config(
    page_title="Emotion Detection using NLP",
    page_icon="🤖",
    layout="wide"
)

# Load Model and Vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Emotion Mapping
emotion_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

emotion_icons = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# Header
st.markdown("""
<h1 style='text-align:center; color:#4F46E5;'>
🤖 Emotion Detection using NLP
</h1>

<p style='text-align:center; font-size:20px;'>
Analyze text and predict emotions using Machine Learning
</p>
""", unsafe_allow_html=True)

# Project Overview
st.info("""
📊 Dataset: 16,000+ Emotion-labelled Text Samples

⚙️ Feature Extraction: TF-IDF Vectorization

🧠 Machine Learning Model for Emotion Classification

🚀 Deployed on Streamlit Cloud
""")

# Input Section
st.markdown("### ✍️ Enter Your Text")

user_input = st.text_area(
    "",
    height=150,
    placeholder="Example: I am very excited about my new project!"
)

# Prediction
if st.button("🔮 Predict Emotion"):

    if user_input:

        text = user_input.lower()

        text_vec = vectorizer.transform([text])

        prediction = model.predict(text_vec)

        emotion = emotion_map[prediction[0]]

        st.success(
            f"{emotion_icons[emotion]} Predicted Emotion: {emotion.upper()}"
        )

    else:
        st.warning("⚠️ Please enter some text.")

# Sample Inputs
st.markdown("### 🎭 Try These Examples")

col1, col2, col3 = st.columns(3)

with col1:
    st.code("I am feeling amazing today")

with col2:
    st.code("I feel lonely and depressed")

with col3:
    st.code("I am scared about tomorrow")

# About Project
st.markdown("---")

st.markdown("""
### 🚀 About This Project

This project uses Natural Language Processing (NLP)
and Machine Learning to identify emotions from text.

✔ Text Preprocessing

✔ TF-IDF Vectorization

✔ Emotion Classification

✔ Streamlit Deployment

✔ Real-time Prediction
""")

# Footer
st.markdown("---")

st.markdown("""
<div style='text-align:center'>
Built with ❤️ by <b>Rutuja Bhandari</b><br>
AIML Student | Exploring NLP & Machine Learning
</div>
""", unsafe_allow_html=True)