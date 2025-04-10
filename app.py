import os
import json
import datetime
import random
import firebase_admin
from firebase_admin import credentials, firestore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq
import streamlit as st

# -------------------------------
# Firebase Setup
# -------------------------------

# Load Firebase credentials
cred = credentials.Certificate("firebase key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------------------
# Constants and Configs
# -------------------------------

UNIVERSITY_RESOURCES = {
    "Centennial College": "Visit the Student Wellness Centre: https://www.centennialcollege.ca/student-health",
    "University of Toronto": "Check U of T’s mental health services: https://mentalhealth.utoronto.ca/",
}

MOTIVATIONAL_QUOTES = [
    "Stay focused! Every small step brings you closer to success. 💪",
    "You’re capable of amazing things—keep pushing forward!",
    "Don’t let stress take over! Take breaks, breathe, and keep going. 🚀"
]

# -------------------------------
# Helper Functions
# -------------------------------

def save_to_firestore(user_id, data):
    db.collection("users").document(user_id).set(data, merge=True)

def get_from_firestore(user_id):
    doc = db.collection("users").document(user_id).get()
    return doc.to_dict() if doc.exists else {}

def update_user_data(user_id, key, value):
    data = get_from_firestore(user_id)
    data[key] = value
    save_to_firestore(user_id, data)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)["compound"]
    if score <= -0.5:
        return "sad"
    elif -0.5 < score <= -0.1:
        return "frustrated"
    elif -0.1 < score < 0.1:
        return "neutral"
    elif 0.1 <= score < 0.5:
        return "happy"
    else:
        return "excited"

def load_llm():
    return ChatGroq(
        temperature=0.2,
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3-70b-8192"
    )

def generate_summary(convo_history, llm):
    prompt = f"You are a helpful assistant. Summarize this conversation in 1-2 sentences:\n{convo_history}"
    return llm.invoke(prompt).content.strip()

def check_deadlines(user_id):
    data = get_from_firestore(user_id)
    deadlines = data.get("deadlines", {})
    today = datetime.date.today()
    upcoming = [
        task for task, date in deadlines.items()
        if datetime.date.fromisoformat(date) <= today + datetime.timedelta(days=3)
    ]
    if upcoming:
        return f"Reminder! Upcoming deadlines: {', '.join(upcoming)}."
    return "No major deadlines soon. Keep it up!"

def send_daily_motivation():
    return random.choice(MOTIVATIONAL_QUOTES)

# -------------------------------
# Streamlit App Interface
# -------------------------------

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Chatbot for Students")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_id" not in st.session_state:
    st.session_state.user_id = ""

with st.sidebar:
    st.header("👤 Set Up Profile")
    name = st.text_input("Name")
    major = st.text_input("Major")
    year = st.text_input("Year of Study")
    stressors = st.text_area("Common Stressors")
    university = st.text_input("University")
    if st.button("Save Profile"):
        if name:
            st.session_state.user_id = name
            update_user_data(name, "name", name)
            update_user_data(name, "major", major)
            update_user_data(name, "year_of_study", year)
            update_user_data(name, "common_stressors", stressors)
            update_user_data(name, "university", university)
            st.success("Profile saved!")
        else:
            st.warning("Please enter a valid name.")

st.subheader("💬 Chat")

message = st.text_input("You:", key="user_input")
if st.button("Send") and message and st.session_state.user_id:
    user_id = st.session_state.user_id
    user_data = get_from_firestore(user_id)
    major = user_data.get("major", "student")
    emotion = analyze_sentiment(message)
    update_user_data(user_id, "last_emotion", emotion)

    llm = load_llm()
    convo = "\n".join([f"User: {u}\nBot: {b}" for u, b in st.session_state.chat_history])
    system_prompt = (
        f"You are a supportive assistant helping a {major} student. Keep your messages readable and don't make them too long.\n"
        f"Name: {user_id}\nLast emotion: {emotion}\n"
        f"Last conversation: {user_data.get('last_conversation', '')}\n"
        f"Offer helpful advice and mental wellness support."
    )
    full_prompt = f"{system_prompt}\nUser: {message}"
    response = llm.invoke(full_prompt).content.strip()

    st.session_state.chat_history.append((message, response))
    update_user_data(user_id, "last_conversation", generate_summary(convo + f"\nUser: {message}\nBot: {response}", llm))

# Display conversation history
for i, (usr, bot) in enumerate(st.session_state.chat_history[::-1]):
    st.markdown(f"**You:** {usr}")
    st.markdown(f"**Bot:** {bot}")
    st.markdown("---")
