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
logging.basicConfig(level=logging.info,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

# Function to retrieve the openai key
def get_openai_api_key() -> Optional[str]:
    """
    Retrieve the OpenAI api key from the environment variables or Streamlit secrets

    Returns:
        Optional[str]: The OpenAI API key if found, None otherwise
    """
    return os.getenv("OPENAI_API_KEY") or st.secrets("OPENAI_API_KEY")

# OpenAI api key setup
OPENAI_API_KEY = get_openai_api_key()

if not OPENAI_API_KEY:
    st.error(ERROR_API_KEY_NOT_FOUND)
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


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

def get_knight_response(user_input: str) -> str:
    """
    Get the knight's response using LangChain.

    Args:
        user_input (str): The user's input message.

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
    llm = ChatOpenAI(temperature=0.7)

    chain= prompt | llm

    try:
        response = chain.invoke({
            "chat_history": st.session_state.conversation_memory.load_memory_variables({})["chat_history"],
            "human_input": user_input,
        })
        st.session_state.conversation_memory.save_context({"input": user_input}, {"output": response.content})
        return response.content

    except Exception as e:
        logger.error(f"Error in getting the knight's words or Knight didn't wish to respond: str(e)")
        return ERROR_RESPONSE_GENERATION

def main():

    initialize_session_state()

    # Title of the Webpage
    st.title(":orange[Campsite] of :blue[the Resting Knight] :fire:")

    # Image of Pixel knight
    st.image("/workspaces/pixel-talk/docs/knight.png", use_column_width=True)

    if st.button("Start Chat with Knight"):
        st.session_state.chat_started = True
        st.session_state.music_playing = True

    if st.session_state.chat_started:
        user_input = st.text_input("Your message to the Knight:")
        if user_input:
            # Placeholder for knight's response
            st.text_area("Knight says:", value="Greetings, traveler!", height=100)

    # if st.session_state.music_playing:
    #     play_background_music('/workspaces/pixel-talk/docs/knights_reflection.mp3')

if __name__ == "__main__":
    main()