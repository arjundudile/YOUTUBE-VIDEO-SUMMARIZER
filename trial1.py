import streamlit as st
import yt_dlp
from moviepy.editor import AudioFileClip
import whisper
from transformers import pipeline
import os
import gc

# Function to download video using yt-dlp
def download_video_with_yt_dlp(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': './downloads/%(title)s.%(ext)s',  # Save in a specific folder
        'retries': 10,                              # Retry up to 10 times
        'fragment_retries': 10,                     # Retry individual fragments
        'socket_timeout': 30,                       # Timeout for network operations
        'verbose': True                             # Enable verbose logs for debugging
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            audio_file_path = ydl.prepare_filename(info_dict)
            return audio_file_path
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

# Function to process audio file
def process_audio(audio_path):
    if audio_path is None or not os.path.exists(audio_path):
        st.error(f"Error: The audio file was not found at {audio_path}")
        return None

    audio_clip = AudioFileClip(audio_path)
    output_audio = "processed_audio.mp3"  # Save as mp3 or another format
    audio_clip.write_audiofile(output_audio)
    return output_audio

# Function to transcribe audio using Whisper (with language option)
def transcribe_audio(audio_file, language=None):
    try:
        # Whisper supports different models; you can experiment with 'base', 'medium', or 'large' based on your needs.
        whisper_model = "base"  # Default to base model
        if language == 'fr':  # French language might benefit from a larger model
            whisper_model = "medium"  # You can try changing to "large" for even better accuracy
        elif language != 'en':  # For other languages, you can also use a larger model
            whisper_model = "medium"

        # Load the appropriate Whisper model
        model = whisper.load_model(whisper_model)

        # Transcribe the audio file
        result = model.transcribe(audio_file, language=language)  # Let Whisper auto-detect language if None
        return result['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# Function to summarize transcribed text (supports multiple languages)
def summarize_text(text, language='en'):
    if language == 'en':
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    else:
        summarizer = pipeline("summarization", model="facebook/mbart-large-50-many-to-one-mmt")
    
    summary = summarizer(text, max_length=150, min_length=100, do_sample=False)
    return summary[0]['summary_text']

# Streamlit App Layout
st.title("YouTube Video Summarizer")

# Sidebar options
option = st.sidebar.radio(
    "Choose an option:",
    ("Transcribe", "Summarize", "Both"),
    index=2  # Default to "Both"
)

# Language selection for transcription and summarization
language = st.sidebar.selectbox("Select Language for Transcription:", ["en", "es", "fr", "de", "it", "pt", "ja", "zh", "hi"])

# Video URL Input
video_url = st.text_input("Enter YouTube Video URL:")

# Process Button
if st.button("Process"):
    if video_url:
        with st.spinner("Processing..."):
            # Step 1: Download video and extract audio
            audio_path = download_video_with_yt_dlp(video_url)
            st.success("Audio downloaded!")

            # Step 2: Process the audio
            processed_audio_path = process_audio(audio_path)
            if processed_audio_path:
                st.success(f"Processed audio file saved at: {processed_audio_path}")
            else:
                st.error("Audio processing failed.")
                st.stop()

            # Option-specific processing
            if option in ("Transcribe", "Both"):
                # Step 3: Transcribe the audio
                transcribed_text = transcribe_audio(processed_audio_path, language)
                if transcribed_text:
                    st.success("Transcription completed!")
                    st.text_area("Transcribed Text:", transcribed_text, height=200)
                else:
                    st.error("Transcription returned no text.")
                    st.stop()

            if option in ("Summarize", "Both") and 'transcribed_text' in locals():
                # Step 4: Summarize the transcribed text
                if transcribed_text:
                    summary = summarize_text(transcribed_text[:1000], language)  # Limit to the first 1000 characters
                    st.subheader("Summary:")
                    st.write(summary)
                else:
                    st.warning("No text available for summarization.")

        st.balloons()  # Show animated balloons when processing is completed
        st.success("Processing completed! 🎉")
    else:
        st.warning("Please enter a video URL.")
