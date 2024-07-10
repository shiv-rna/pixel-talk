import streamlit as st
import base64

def main():
    # Title of the Webpage
    st.title(":orange[Campsite] of :blue[_Resting _knight] :fire:")

    # Image of Pixel knight
    st.image("/workspaces/pixel-talk/docs/knight.png", use_column_width=True)

    if st.button("Start Chat with Knight"):
        st.session_state.chat_started = True

    if st.session_state.get('chat_started', False):
        user_input = st.text_input("Your message to the Knight:")
        if user_input:
            # Placeholder for knight's response
            st.text_area("Knight says:", value="Greetings, traveler!", height=100)

        
    # Background music controls
    st.sidebar.title("Music Controls")
    
    # Music toggle
    if 'music_playing' not in st.session_state:
        st.session_state.music_playing = False

    if st.sidebar.button("Toggle Background Music"):
        st.session_state.music_playing = not st.session_state.music_playing

    # Display current music state
    st.sidebar.write(f"Music is {'playing' if st.session_state.music_playing else 'paused'}")

    # Add background music (if playing)
    if st.session_state.music_playing:
        audio_file = open('background_music.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.markdown(
            f'<audio autoplay loop><source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
