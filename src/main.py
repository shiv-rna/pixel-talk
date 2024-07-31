import streamlit as st
import base64
import logging
import os
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Importing constants from config file
from config import (
    IMAGE_PATH, AUDIO_PATH, TITLE, START_CHAT_BUTTON_TEXT, USER_MESSAGE_PROMPT,
    ERROR_API_KEY_NOT_FOUND, ERROR_AUDIO_NOT_FOUND, ERROR_IMAGE_NOT_FOUND,
    ERROR_RESPONSE_GENERATION
)

# Setup basic Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)


def initialize_session_state() -> None:
    """
    Initialize session state variables.
    
    This function sets up the initial state for the Streamlit app,
    including chat status, music playing status, conversation memory,
    and message history
    """
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False
    if 'music_playing' not in st.session_state:
        st.session_state.music_playing = False
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if 'messages' not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []


def manage_api_key() -> Optional[str]:
    """
    Manage the OpenAI API key input and storage

    Returns:
        Optional[str]: The OpenAI API key if provided, None otherwise
    """
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None

    api_key = st.sidebar.text_input("Enter your OpenAI API key🔑:", type="password", key="openai_api_key")
    if api_key:
        st.session_state.open_api_key = api_key

    return st.session_state.openai_api_key

def play_background_music(file_path) -> None:
    """
    Embed and play background music in the Streamlit app.

    Args:
        file_path (str): The path to the audio file
    """
    try:
        with open(file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.markdown(
                f'<audio autoplay loop><source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3"></audio>',
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file_path}")
        st.error(ERROR_AUDIO_NOT_FOUND)

def get_knight_response(user_input: str, api_key: str) -> str:
    """
    Get the knight's response using LangChain.

    Args:
        user_input (str): The user's input message.
        api_key (str): The OpenAI api key

    Returns:
        str: The knight's response
    """
    template= """
    You are a wise and knowledgeable knight who always responds in medieval style
    Your task is to provide top 5 news highlists from around the world of the user provided month and year

    Current conversation:
    {chat_history}
    Human: {human_input}
    Knight:
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.7, api_key=api_key)
    chain= prompt | llm

    try:
        response = chain.invoke({
            "chat_history": st.session_state.conversation_memory.load_memory_variables({})["chat_history"],
            "human_input": user_input,
        })
        st.session_state.conversation_memory.save_context({"input": user_input}, {"output": response.content})
        return response.content

    except Exception as e:
        logger.error(f"Silence was observed in knight's words or Knight didn't wish to respond: str(e)")
        return ERROR_RESPONSE_GENERATION

def display_chat_messages()-> None:
    """
    Display chat messages in the Streamlit application
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(api_key: str) -> None:
    """
    Handle user input and get knight's response
    
    Args:
        api_key (str): The OpenAI api key
    """
    user_input = st.chat_input(USER_MESSAGE_PROMPT)
    if user_input:
        st.session_state.messages.append({"role":"user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        knight_response = get_knight_response(user_input, api_key)
        st.session_state.messages.append({"role":"assistant", "content":knight_response})
        with st.chat_message("assistant"):
            st.markdown(knight_response)


def main() -> None:
    """
    Main function to run the streamlit app.
    """

    initialize_session_state()

    # Title of the Webpage
    st.title(TITLE)

    # Image of Pixel knight
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, use_column_width=True)
    else:
        logger.warning(f"Image file for the knight not found: {IMAGE_PATH}")
        st.error(ERROR_IMAGE_NOT_FOUND)

    # Check for api key before allowing the chat to initiate
    api_key=manage_api_key()

    if not api_key:
        st.warning("Please enter OpenAI API key in the sidebar to initiate the chat")
    else:
        if st.button(START_CHAT_BUTTON_TEXT):
            st.session_state.chat_started = True
            st.session_state.music_playing = True

        if st.session_state.chat_started:
            display_chat_messages()
            handle_user_input(api_key)

        if st.session_state.music_playing:
            play_background_music(AUDIO_PATH)

if __name__ == "__main__":
    main()