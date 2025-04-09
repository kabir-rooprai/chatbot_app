import streamlit as st
import os
import json
import random
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq

# -------------------------------
# Constants and Setup
# -------------------------------

USER_DATA_FILE = "user_data.json"

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

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

def save_user_data(user_data):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(user_data, file, indent=4)

def update_user_data(user_id, key, value):
    user_data = load_user_data()
    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id][key] = value
    save_user_data(user_data)

def update_student_profile(user_id, major, year_of_study, common_stressors, university):
    user_data = load_user_data()
    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id]["major"] = major
    user_data[user_id]["year_of_study"] = year_of_study
    user_data[user_id]["common_stressors"] = common_stressors
    user_data[user_id]["university"] = university
    save_user_data(user_data)

def get_mental_health_resources(user_id):
    user_data = load_user_data()
    university = user_data.get(user_id, {}).get("university", "your university")
    return UNIVERSITY_RESOURCES.get(university, "Please check your university’s student wellness resources.")

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

def check_deadlines(user_id):
    user_data = load_user_data()
    deadlines = user_data.get(user_id, {}).get("deadlines", {})
    today = datetime.date.today()
    upcoming = [
        task for task, date in deadlines.items()
        if datetime.date.fromisoformat(date) <= today + datetime.timedelta(days=3)
    ]
    if upcoming:
        return f"Reminder! You have upcoming deadlines: {', '.join(upcoming)}. Don’t forget to plan ahead!"
    return None

def send_daily_motivation():
    return random.choice(MOTIVATIONAL_QUOTES)

def load_llm():
    return ChatGroq(
        temperature=0.2,
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile"
    )

def generate_summary(convo, llm):
    prompt = f"""You are a helpful assistant. Summarize this conversation in 1-2 sentences.
Conversation: {' '.join(convo)}
Summary:"""
    return llm.invoke(prompt).content.strip()

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("💬 Mental Health Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

# -------------------------------
# Profile Setup
# -------------------------------

with st.expander("📋 Set Up Your Profile", expanded=not st.session_state.show_chat):
    name = st.text_input("Your Name")
    major = st.text_input("Your Major")
    year = st.text_input("Year of Study")
    stressors = st.text_input("Common Stressors")
    university = st.text_input("University")
    if st.button("✅ Save Profile"):
        if name.strip():
            update_user_data(name, "name", name)
            update_student_profile(name, major, year, stressors, university)
            st.session_state.user_id = name
            st.success(f"Profile saved for {name}")
            st.session_state.show_chat = True
        else:
            st.error("Please enter a valid name to continue.")

# -------------------------------
# Chat UI and Logic
# -------------------------------

if st.session_state.show_chat:
    user_id = st.session_state.user_id
    user_data = load_user_data()
    llm = load_llm()
    history = st.session_state.chat_history

    st.divider()
    st.subheader("🧠 Chat with Your Supportive Bot")

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("How are you feeling today?")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        history.append({"role": "user", "content": user_input})

        emotion = analyze_sentiment(user_input)
        update_user_data(user_id, "last_emotion", emotion)

        context = (
            f"You are a compassionate mental health assistant for a student named {user_id}, studying {user_data.get(user_id, {}).get('major', 'a subject')} Try to keep your messages relatively short and readble.\n"
            f"Last known emotion: {user_data[user_id].get('last_emotion', 'None')}\n"
            f"Last conversation summary: {user_data[user_id].get('last_conversation', 'None')}\n"
            f"Respond with empathy and offer coping tips or helpful resources.\n"
        )

        prompt = context
        for msg in history[-4:]:  # last 2 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"

        try:
            response = llm.invoke(prompt).content.strip()
        except Exception as e:
            response = f"Sorry, an error occurred: {e}"

        with st.chat_message("assistant"):
            st.markdown(response)
        history.append({"role": "assistant", "content": response})

        summary = generate_summary([m["content"] for m in history if m["role"] == "user"], llm)
        update_user_data(user_id, "last_conversation", summary)

        deadline_msg = check_deadlines(user_id)
        if deadline_msg:
            st.info(deadline_msg)

        if "stress" in user_input.lower() or emotion in ["sad", "frustrated"]:
            st.info(send_daily_motivation())

        st.toast(get_mental_health_resources(user_id))
