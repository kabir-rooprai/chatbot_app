import streamlit as st
import datetime
import random
import json
import firebase_admin
from firebase_admin import credentials, firestore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq

# -------------------------------
# Load secrets
# -------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
FIREBASE_CONFIG = st.secrets["FIREBASE"]

cred = credentials.Certificate({
    "type": FIREBASE_CONFIG["type"],
    "project_id": FIREBASE_CONFIG["project_id"],
    "private_key_id": FIREBASE_CONFIG["private_key_id"],
    "private_key": FIREBASE_CONFIG["private_key"],
    "client_email": FIREBASE_CONFIG["client_email"],
    "client_id": FIREBASE_CONFIG["client_id"],
    "auth_uri": FIREBASE_CONFIG["auth_uri"],
    "token_uri": FIREBASE_CONFIG["token_uri"],
    "auth_provider_x509_cert_url": FIREBASE_CONFIG["auth_provider_x509_cert_url"],
    "client_x509_cert_url": FIREBASE_CONFIG["client_x509_cert_url"],
    "universe_domain": FIREBASE_CONFIG["universe_domain"],
})

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------------
# Helper Functions
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

def load_user_data(user_id):
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {}

def save_user_data(user_id, data):
    db.collection("users").document(user_id).set(data)

def update_user_data(user_id, key, value):
    user_data = load_user_data(user_id)
    user_data[key] = value
    save_user_data(user_id, user_data)

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
    return "excited"

def load_llm():
    return ChatGroq(
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

def check_deadlines(user_data):
    today = datetime.date.today()
    deadlines = user_data.get("deadlines", {})
    upcoming = [
        task for task, date in deadlines.items()
        if datetime.date.fromisoformat(date) <= today + datetime.timedelta(days=3)
    ]
    return f"📌 Upcoming deadlines: {', '.join(upcoming)}" if upcoming else "✅ No major deadlines soon."

def get_resources(university):
    return UNIVERSITY_RESOURCES.get(
        university, 
        "ℹ️ Please check your university's website for wellness resources."
    )

# -------------------------------
# Streamlit App Layout
# -------------------------------

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Chatbot for Students")

with st.sidebar:
    st.header("👤 Your Profile")
    user_id = st.text_input("Name (used as ID)")
    major = st.text_input("Major")
    year = st.text_input("Year of Study")
    stressors = st.text_input("Common Stressors")
    university = st.text_input("University")
    if st.button("💾 Save Profile"):
        update_user_data(user_id, "name", user_id)
        update_user_data(user_id, "major", major)
        update_user_data(user_id, "year", year)
        update_user_data(user_id, "common_stressors", stressors)
        update_user_data(user_id, "university", university)
        st.success("✅ Profile saved successfully!")

if user_id:
    st.subheader(f"Hello {user_id}, how are you feeling today?")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg[0])
        with st.chat_message("assistant"):
            st.markdown(msg[1])

    user_input = st.chat_input("Type your message here...")
    if user_input:
        emotion = analyze_sentiment(user_input)
        update_user_data(user_id, "last_emotion", emotion)

        llm = load_llm()
        context = load_user_data(user_id)
        convo_history = "\n".join([f"User: {u}\nBot: {b}" for u, b in st.session_state.chat_history])

        prompt = (
            f"You are a warm and supportive chatbot helping students with mental health.\n"
            f"User info: {context}\n"
            f"Previous emotion: {emotion}\n"
            f"Chat history:\n{convo_history}\n"
            f"User: {user_input}\n"
            f"Respond empathetically and helpfully in 1-3 sentences."
        )

        bot_response = llm.invoke(prompt).content.strip()
        st.session_state.chat_history.append((user_input, bot_response))
        update_user_data(user_id, "last_conversation", bot_response)

        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(bot_response)

    st.divider()
    st.markdown(get_resources(university if university else ""))
    st.markdown(check_deadlines(load_user_data(user_id)))
    st.markdown(f"💬 *Motivation:* {random.choice(MOTIVATIONAL_QUOTES)}")
