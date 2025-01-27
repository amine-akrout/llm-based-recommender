import os
import time

import requests
import streamlit as st

# Backend API URL
API_URL = os.getenv("API_URL", "http://localhost:8000/recommend")

# Page Config
st.set_page_config(
    page_title="👗 AI Fashion Recommender", page_icon="🛍️", layout="centered"
)

# Custom CSS for Fancy UI with Padding & Avatars
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        border-radius: 12px;
    }

    .user-message {
        background-color: #007bff;
        color: white;
        padding: 12px;
        border-radius: 10px;
        max-width: 75%;
        text-align: right;
        float: right;
        clear: both;
        margin: 10px 0;
    }

    .bot-message {
        background-color: #f1f1f1;
        color: black;
        padding: 12px;
        border-radius: 10px;
        max-width: 75%;
        text-align: left;
        float: left;
        clear: both;
        margin: 10px 0;
    }

    .chat-wrapper {
        padding: 20px 100px;
    }

    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page Title
st.title("👗 AI Fashion Recommender")
st.write(
    "💡 Ask me about fashion recommendations, and I'll help you find the perfect outfit!"
)

# Chat Display
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

chat_container = st.container()

# Display Chat History
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-message">
                    <img src="https://img.icons8.com/fluency/48/user-male-circle.png" class="avatar"> 
                    {msg["content"]}
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="bot-message">
                    <img src="https://img.icons8.com/?size=100&id=QIRhukOe1BpC&format=png&color=000000" class="avatar"> 
                    {msg["content"]}
                </div>
            """,
                unsafe_allow_html=True,
            )
        st.write("")

st.markdown("</div>", unsafe_allow_html=True)  # Close padding wrapper

# User Input
query = st.text_input("Type your fashion query here...", key="query")

# Fancy Button
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    send_button = st.button("🛍️ Get Recommendation")

# Handle User Submission
if send_button:
    if query:
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message instantly
        with chat_container:
            st.markdown(
                f"""
                <div class="user-message">
                    <img src="https://img.icons8.com/fluency/48/user-male-circle.png" class="avatar"> 
                    {query}
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.write("")

        # Simulate Bot Thinking Effect
        with chat_container:
            st.markdown(
                '<div class="bot-message"><b>🤖 Thinking...</b></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            time.sleep(1.5)  # Simulated delay

        try:
            # Call FastAPI Backend
            response = requests.post(API_URL, json={"question": query})

            if response.status_code == 200:
                answer = response.json().get("answer", "No recommendation found.")
            else:
                answer = "Error: Unable to fetch recommendation."

        except Exception as e:
            answer = f"Error: {str(e)}"

        # Display Full Bot Response Instantly
        with chat_container:
            st.markdown(
                f"""
                <div class="bot-message">
                    <img src="https://img.icons8.com/color/48/chatbot.png" class="avatar"> 
                    {answer}
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.write("")

        # Append bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
