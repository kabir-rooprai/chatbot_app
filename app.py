import streamlit as st
import datetime
import random
import json
import firebase_admin
from firebase_admin import credentials, firestore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq

# --- Load secrets ---
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
    "universe_domain": FIREBASE_CONFIG["universe_domain"]
})

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Constants & Utilities ---
UNIVERSITY_RESOURCES = {
    "Centennial College": "https://www.centennialcollege.ca/student-health",
    "University of Toronto": "https://mentalhealth.utoronto.ca/",
}

MOTIVATIONAL_QUOTES = [
    "Stay focused! Every small step brings you closer to success. 💪",
    "You’re capable of amazing things—keep pushing forward!",
    "Don’t let stress take over! Take breaks, breathe, and keep going. 🚀"
]

def load_user_data(user_id):
    doc = db.collection("users").document(user_id).get()
    return doc.to_dict() if doc.exists else {}

def save_user_data(user_id, data):
    db.collection("users").document(user_id).set(data)

def update_user_data(user_id, key, value):
    data = load_user_data(user_id)
    data[key] = value
    save_user_data(user_id, data)

def analyze_sentiment(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    return (
        "sad" if score <= -0.5 else
        "frustrated" if score <= -0.1 else
        "neutral" if score < 0.1 else
        "happy" if score < 0.5 else
        "excited"
    )

def load_llm():
    return ChatGroq(
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

def generate_summary(history, llm):
    prompt = f"Summarize this chat in 1-2 sentences:\n{history}"
    return llm.invoke(prompt).content.strip()

def get_resources(university):
    return UNIVERSITY_RESOURCES.get(university, "Please check your university’s wellness page.")

# --- Streamlit UI ---
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Chatbot for Students")

st.sidebar.header("👤 User Profile")
user_id = st.sidebar.text_input("Enter your name (acts as ID):")
if user_id:
    major = st.sidebar.text_input("Your Major")
    year = st.sidebar.text_input("Year of Study")
    stressors = st.sidebar.text_input("Common Stressors")
    university = st.sidebar.text_input("University")

    if st.sidebar.button("Save Profile"):
        update_user_data(user_id, "name", user_id)
        update_user_data(user_id, "major", major)
        update_user_data(user_id, "year", year)
        update_user_data(user_id, "common_stressors", stressors)
        update_user_data(user_id, "university", university)
        st.sidebar.success("Profile Saved!")

# --- Chat Interface ---
if user_id:
    st.subheader(f"Hi {user_id}, how are you feeling today?")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    input_col, button_col = st.columns([5, 1])
    with input_col:
        user_input = st.text_input("Type your message:", value="", key="input", label_visibility="collapsed")
    with button_col:
        send_clicked = st.button("Send")

    if send_clicked and user_input:
        emotion = analyze_sentiment(user_input)
        update_user_data(user_id, "last_emotion", emotion)

        llm = load_llm()
        context = load_user_data(user_id)
        convo_history = " ".join([f"User: {u} Bot: {b}" for u, b in st.session_state.chat_history])
        prompt = (
            f"You are a mental health chatbot for students.\n"
            f"User profile: {context}\n"
            f"Emotion: {emotion}\n"
            f"History: {convo_history}\n"
            f"User says: {user_input}\n"
            "Respond naturally and supportively. Keep the message under 200 words."
        )
        bot_response = llm.invoke(prompt).content.strip()
        st.session_state.chat_history.append((user_input, bot_response))
        update_user_data(user_id, "last_conversation", bot_response)
        st.experimental_rerun()

    # Display chat with icons
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/3177/3177440.png' width='30' style='margin-right: 10px; margin-top: 3px;'>
            <div><strong>You:</strong> {user_msg}</div>
        </div>
        <div style='display: flex; align-items: flex-start; margin-bottom: 20px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/4712/4712107.png' width='30' style='margin-right: 10px; margin-top: 3px;'>
            <div><strong>Bot:</strong> {bot_msg}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("End Conversation"):
        if st.session_state.chat_history:
            full_history = " ".join([f"User: {u} Bot: {b}" for u, b in st.session_state.chat_history])
            summary = generate_summary(full_history, load_llm())
            update_user_data(user_id, "conversation_summary", summary)
            st.session_state.chat_history.append((
                "exit", "Thanks for chatting. You're doing amazing. Come back anytime! 🌟"
            ))
            st.success("Conversation saved.")
            st.experimental_rerun()

    st.markdown("---")
    st.markdown(get_resources(university if university else ""))
    st.markdown(f"💬 *Motivation:* {random.choice(MOTIVATIONAL_QUOTES)}")
