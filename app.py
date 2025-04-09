import streamlit as st
import json
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq

# Load API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Load and save user data
USER_DATA_FILE = "user_data.json"

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)["compound"]
    if score <= -0.5:
        return "sad"
    elif score <= -0.1:
        return "frustrated"
    elif score < 0.1:
        return "neutral"
    elif score < 0.5:
        return "happy"
    return "excited"

def load_llm():
    return ChatGroq(
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

st.title("🧠 Mental Health Chatbot")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Profile setup
with st.expander("👤 Set up your profile"):
    name = st.text_input("Your Name")
    major = st.text_input("Major")
    year = st.text_input("Year")
    stressors = st.text_input("Common Stressors")
    university = st.text_input("University")
    if st.button("Save Profile"):
        user_data = load_user_data()
        user_data[name] = {
            "major": major,
            "year": year,
            "stressors": stressors,
            "university": university
        }
        save_user_data(user_data)
        st.session_state.user_id = name
        st.success(f"Profile saved for {name}!")

# Chat section
if st.session_state.user_id:
    user_input = st.text_input("Say something...", key="input")
    if st.button("Send"):
        llm = load_llm()
        chat_prompt = ""
        for turn in st.session_state.history:
            chat_prompt += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
        chat_prompt += f"User: {user_input}\nAssistant:"
        response = llm.invoke(chat_prompt).content.strip()

        st.session_state.history.append({"user": user_input, "bot": response})

    # Display chat history
    for msg in st.session_state.history:
        st.markdown(f"**You:** {msg['user']}")
        st.markdown(f"**Bot:** {msg['bot']}")
else:
    st.warning("Please set up your profile first.")
