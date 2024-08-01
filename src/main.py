import streamlit as st
import base64
import logging
import os
from typing import List, Dict, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from openai import AuthenticationError, RateLimitError

# Importing constants from config file
from config import (
    IMAGE_PATH, AUDIO_PATH, TITLE, START_CHAT_BUTTON_TEXT, USER_MESSAGE_PROMPT,
    ERROR_AUDIO_NOT_FOUND, ERROR_IMAGE_NOT_FOUND
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
    default_values = {
        'chat_started': False,
        'music_playing':False,
        'conversation_memory': ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        'messages': [],
        'validated_api_key': False,
        'last_validated_key': None,
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data
def load_image(path: str) -> Optional[str]:
    """
    Load and cache the image
    """
    if os.path.exists(path):
        return path
    logger.warning(f"Image file not found: {path}")
    return None

@st.cache_data
def load_audio(file_path: str) -> Optional[str]:
    """
    Load and cache the audio file
    """
    try:
        with open(file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        return base64.b64encode(audio_bytes).decode()
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading audio file: {str(e)}")
        return None

@st.cache_data
def validate_api_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate the OpenAI API key.
    
    Args:
        api_key (str): The OpenAI API key to be validated.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether the API key is valid,
                                     and an optional error message if the API key is invalid.
    """
    try:
        ChatOpenAI(temperature=0.7, api_key=api_key)
        return True, None
    except AuthenticationError:
        return False, "Invalid API key. Please check & try again"
    except Exception as e:
        return False, f"An error occurred while validating the API key: {str(e)}"
    

def manage_api_key() -> Optional[str]:
    """
    Manage the OpenAI API key input and storage

    Returns:
        Optional[str]: The OpenAI API key if provided, None otherwise
    """
    api_key = st.sidebar.text_input("Enter your OpenAI API key🔑:", type="password", key="api_key_input")

    if api_key:
        if not st.session_state.validated_api_key or api_key != st.session_state.last_validated_key:
            is_valid, error_message = validate_api_key(api_key)
            if is_valid:
                st.session_state.validated_api_key = True
                st.session_state.last_validated_key = api_key
                st.sidebar.success("API key is valid ✅")
                return api_key
            else:
                st.sidebar.error(error_message)
                logger.error(f"API key validation error: {error_message}")
                st.session_state.validated_api_key = False
                st.session_state.last_validated_key = None
                return None
        elif st.session_state.validated_api_key:
            return api_key
    
    return None


def play_background_music(audio_base64: str) -> None:
    """
    Embed and play background music in the Streamlit app.

    Args:
        audio_base64 (str): The path to the audio file
    """
    st.markdown(
        f'<audio autoplay loop><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True
            )

@st.cache_resource
def get_llm_chain(_api_key: str):
    """
    Get the LLM chain for generating responses.
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
    llm = ChatOpenAI(temperature=0.7, api_key=_api_key)
    return prompt | llm

def get_knight_response(user_input: str, api_key: str) -> str:
    """
    Get the knight's response using LangChain.

    Args:
        user_input (str): The user's input message.
        api_key (str): The OpenAI api key

    Returns:
        str: The knight's response
    """
    chain = get_llm_chain(api_key)
    try:
        response = chain.invoke({
            "chat_history": st.session_state.conversation_memory.load_memory_variables({})["chat_history"],
            "human_input": user_input,
        })
        st.session_state.conversation_memory.save_context({"input": user_input}, {"output": response.content})
        return response.content
    except AuthenticationError:
        return "Authentication error: Invalid API key. Please check your API key and try again"
    except RateLimitError:
        return "Rate limit exceeded. Please wait a moment before trying again"
    except Exception as e:
        logger.error(f"Error in getting the knight's response: {str(e)}")
        return f"An error occured: {str(e)}"

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
    image_path = load_image(IMAGE_PATH)
    if image_path:
        st.image(image_path, use_column_width=True)
    else:
        st.error(ERROR_IMAGE_NOT_FOUND)

    # Check for api key before allowing the chat to initiate
    api_key = manage_api_key()

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
            audio_base64 = load_audio(AUDIO_PATH)
            if audio_base64:
                play_background_music(audio_base64)
            else:
                st.error(ERROR_AUDIO_NOT_FOUND)

if __name__ == "__main__":
    main()