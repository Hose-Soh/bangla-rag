import streamlit as st
from streamlit_chat import message
import requests

# Set page config
st.set_page_config(page_title="Bangla RAG", page_icon="🤖")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_question_to_flask_api(question):
    """
    Sends the user question to the Flask API and retrieves the response.
    """
    try:
        response = requests.post(
            "http://127.0.0.1:5000/agent",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json().get("answer", "No answer returned.")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def chatbot():
    # Main chat container
    chat_container = st.container()
    with chat_container:
        for index, message_data in enumerate(st.session_state.get('messages', [])):
            message(
                message_data["content"],
                is_user=message_data["role"] == "user",
                key=f"message_{index}",
                avatar_style=message_data.get("avatar_style", "no-avatar")
            )

    # User input
    user_input = st.chat_input("How may I assist?")
    
    if user_input:
        # Add user message to UI
        message(user_input, is_user=True, key=f"message_{len(st.session_state.messages)}", avatar_style="no-avatar")

        # Get response from Flask API
        response_text = send_question_to_flask_api(user_input)

        # Add bot response to UI
        message(response_text, is_user=False, key=f"message_{len(st.session_state.messages) + 1}", avatar_style="no-avatar")

        # Update message history
        st.session_state.messages.append({"role": "user", "content": user_input, "avatar_style": "no-avatar"})
        st.session_state.messages.append({"role": "bot", "content": response_text, "avatar_style": "no-avatar"})
        st.rerun()

# Run the chatbot app
chatbot()
