import streamlit as st
import base64

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False
    if 'music_playing' not in st.session_state:
        st.session_state.music_playing = False

def play_background_music(file_path):
    """Embed and play background music in the Streamlit app."""
    with open(file_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.markdown(
            f'<audio autoplay loop><source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )

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

    if st.session_state.music_playing:
        play_background_music('/workspaces/pixel-talk/docs/knights_reflection.mp3')

if __name__ == "__main__":
    main()