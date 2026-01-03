import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.title("Arif's Chatbot")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful AI assistant")
    ]

# Display previous messages
for msg in st.session_state.chat_history[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input (auto-rerun safe)
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user message
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call model
    response = model.invoke(st.session_state.chat_history)

    # Store AI response
    st.session_state.chat_history.append(
        AIMessage(content=response.content)
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)
