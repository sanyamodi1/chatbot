import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create the OpenAI client
client = OpenAI(api_key=api_key)

# Streamlit UI setup
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("💬 Chatbot")
st.markdown("Ask anything related to your course or subject!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant for course-related questions."}
    ]

# Display previous messages
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Prompt input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"❌ Error: {str(e)}"

    # Display and save assistant reply
    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
