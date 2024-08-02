import streamlit as st
import base64
import logging
import os
from typing import List, Dict, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from openai import AuthenticationError, RateLimitError
import pandas as pd
from io import BytesIO

# Importing constants from config file
from config import (
    IMAGE_PATH,
    AUDIO_PATH,
    TITLE,
    START_CHAT_BUTTON_TEXT,
    USER_MESSAGE_PROMPT,
    ERROR_AUDIO_NOT_FOUND,
    ERROR_IMAGE_NOT_FOUND,
)

# Setup basic Logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)


def initialize_session_state() -> None:
    """
    Initialize session state variables for Streamlit app.
    
    This function sets up the initial state for various app components, including:
    - Chat Status
    - Music playing status
    - Conversation memory
    - Message history
    - API key validation status

    This function uses a dictionary of default values and updates the session state
    only for keys that are not already present.
    """
    default_values = {
        "chat_started": False,
        "music_playing":False,
        "conversation_memory": ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        ),
        "messages": [],
        "validated_api_key": False,
        "last_validated_key": None,
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data
def load_image(path: str) -> Optional[str]:
    """
    Load and cache an image file.

    This function attempts to load an image file from the given path.
    If sucessful, it returns the path; otherwise, it logs a warning
    and returns None

    Args:
        path(str): The file path of the image to be loaded.
    
    Returns:
        Optinal[str]: The file path if the image exists, None otherwise
    """
    if os.path.exists(path):
        return path
    logger.warning(f"Image file not found: {path}")
    return None

@st.cache_data
def load_audio(file_path: str) -> Optional[str]:
    """
    Load and cache an audio file, converting it to base64 encoding.

    This function reads an audio file from the given path, encodes it to base64,
    and returns the encoded string. If the file is not found or an error occurs
    during reading, it logs an error and returns None.

    Args:
        file_path(str): The file path of the audio to be loaded.
    
    Returns:
        Optional[str]: Base64 encoded audio data if sucessful, None otherwise
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
    Validate the OpenAI API key

    This function attempts to create a ChatOpenAI instance with the provided API key.
    If successful, it returns True with no error message. If an authentication error
    occurs, it returns False with an error message. For any other exception, it returns
    False with a general error message.
    
    Args:
        api_key (str): The OpenAI API key to be validated.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether 
        the API key is valid, and an optional error message if the API key is invalid.
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
    Manage the OpenAI API key input and validation process.

    This function handles the API key input through the Streamlit sidebar,
    validates the key, and manages the validation state. It returns the
    validated API key if available, or None otherwise.

    Returns:
        Optional[str]: The validated OpenAI API key if provided, None otherwise.
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

    This function creates an HTML audio element with the provided base64-encoded
    audio data and embeds it in the Streamlit app. It also includes a script to
    set the initial volume of the audio.

    Args:
        audio_base64 (str): The base64-encoded audio data.
    """
    st.markdown(
        f"""
        <audio id="background-music" autoplay loop>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <script>
            var audio = document.getElementById("background-music");
            audio.volume = 0.6; // Adjust volume as needed (0.0 to 1.0)
        </script>
        """,
                unsafe_allow_html=True
            )

@st.cache_resource
def get_llm_chain(_api_key: str):
    """
    Get the LLM chain for generating responses.

    This function creates and returns a language model chain using a custom prompt
    template and the ChatOpenAI model. The chain is designed to generate responses
    in a medieval knight style, providing news highlights for a given month and year.

    Args:
        _api_key (str): The OpenAI API key to initialize the ChatOpenAI model.

    Returns:
        Chain: A LangChain chain object that can be used to generate responses.

    """

    template = """
    You are Sir Galahad, a wise and knowledgeable knight of the Round Table, known for your purity and nobility. 
    Your task is to provide the top 5 real-world news highlights from around the world for the user-provided month and year.
    Present this information as if you're recounting tales from your travels across the realm.

    Remember to:
    1. Speak in a medieval style, using appropriate language and expressions.
    2. Ensure all information is based on real-world events.
    3. Offer brief, knightly insights on each piece of news.
    4. Maintain a tone of honor, chivalry, and wisdom in your responses.
    5. If asked about future dates, focus on positivity and motivation, encouraging the user to shape the future.
    6. Only provide information about events that have actually occurred. Do not invent or speculate about news events.

    If the date is in the future:
    1. Acknowledge that the date is yet to come.
    2. Offer words of encouragement and motivation about shaping the future.
    3. Avoid making predictions or inventing news events.

    Current conversation:
    {chat_history}
    Human: {human_input}
    Sir Galahad:
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.7, api_key=_api_key)
    return prompt | llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.conversation_memory.clear()
    st.session_state.chat_started = False



def get_knight_response(user_input: str, api_key: str) -> str:
    """
    Get the knight's response using LangChain.

    This function processes the user input through the LLM chain to generate
    a response in the style of a medieval knight. It handles various exceptions
    that might occur during the process and returns appropriate error messages.

    Args:
        user_input (str): The user's input message.
        api_key (str): The OpenAI api key

    Returns:
        str: The knight's response or an error message if an exception occurs
        
    """
    chain = get_llm_chain(api_key)
    try:
        response = chain.invoke(
            {
                "chat_history": st.session_state.conversation_memory.load_memory_variables(
                    {}
                )[
                    "chat_history"
                ],
            "human_input": user_input,
            }
        )
        st.session_state.conversation_memory.save_context(
            {"input": user_input}, {"output": response.content}
        )
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
    Display chat messages in the Streamlit application.

    This function iterates through the messages stored in the session state
    and displays them in the Streamlit chat interface, differentiating between
    user and assistant (knight) messages.

    """
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        with st.chat_message(last_message["role"]):
            st.markdown(last_message["content"])


def download_transcript():
    df = pd.DataFrame(st.session_state.messages)
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()


def handle_user_input(api_key: str) -> None:
    """
    Handle user input and get knight's response.

    This function manages the chat input process, including:
    - Capturing user input
    - Appending user messages to the chat history
    - Generating and displaying the knight's response
    - Updating the chat interface

    Args:
        api_key (str): The OpenAI API key for generating responses.

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
    Main function to run the Streamlit app.

    This function orchestrates the entire application flow, including:
    - Initializing the session state
    - Setting up the page title and layout
    - Managing API key input and validation
    - Handling chat initiation and user interactions
    - Playing background music
    - Displaying chat messages and handling user input

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
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Camp" if st.session_state.chat_started else START_CHAT_BUTTON_TEXT):
                if st.session_state.chat_started:
                    reset_chat()
                else:
                    st.session_state.chat_started = True
                    st.session_state.music_playing = False

        with col2:
            transcript = download_transcript()
            if transcript:
                st.download_button(
                    label="Download Transcript of Camp",
                    data=transcript,
                    file_name="chat_transcript.csv",
                    mime="text/csv",
                )

        if st.session_state.chat_started:
            if not st.session_state.music_playing:
                audio_base64 = load_audio(AUDIO_PATH)
                if audio_base64:
                    play_background_music(audio_base64)
                     # Counter-intuitively, we set this to False to make the music continue playing.
                    # This ensures the audio element is re-embedded on each Streamlit rerun.
                    st.session_state.music_playing = False 
                else:
                    st.error(ERROR_AUDIO_NOT_FOUND)

            display_chat_messages()
            handle_user_input(api_key)

    

if __name__ == "__main__":
    main()