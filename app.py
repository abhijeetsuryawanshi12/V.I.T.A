
import streamlit as st
import os
import tempfile
import logging
import json
from dotenv import load_dotenv
from audio_processor import AudioProcessor

# --- Page Configuration ---
st.set_page_config(
    page_title="🎙️ Live Transcription App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- CSS for UI enhancements ---
st.markdown("""
<style>
    /* Change the background of the main content area */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #f0f2f6;
    }

    /* Center the main content block and add padding */
    .block-container {
        max-width: 900px;
        margin: auto;
        padding: 2rem 1rem;
    }

    /* Card style for the recorder */
    .recorder-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }

    /* Headers style */
    h1, h2, h3 {
        text-align: center;
        color: #2c3e50; /* Dark blue-grey for headers */
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }

    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #ddd;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.1rem;
    }
    .speaker-name {
        font-weight: bold;
        font-size: 1.2em;
    }
    .timestamp {
        font-size: 0.9em;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching the Audio Processor ---
@st.cache_resource
def get_audio_processor():
    """Load and cache the AudioProcessor instance to avoid reloading models."""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            st.error("Hugging Face token not found. Please set HF_TOKEN in your .env file for speaker diarization.")
            st.stop()
        
        with st.spinner("Loading audio processing models... This may take a minute on first run."):
            processor = AudioProcessor(
                whisper_model_size="large",
                auth_token=hf_token,
                compute_type="float32" # Use "int8" for faster CPU performance
            )
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize AudioProcessor: {e}")
        st.error(f"Error initializing audio processor: {e}")
        st.stop()

# --- Callback for audio recorder ---
def on_new_recording():
    """Callback function to handle new audio recordings."""
    # When new audio is recorded, reset the transcript and set a flag to process.
    if st.session_state.recorder is not None:
        st.session_state.audio_processed = False
        st.session_state.transcript_data = []

# --- Main App Logic ---
def main():
    st.title("🎙️ Live Audio Transcription & Diarization")
    st.markdown("<p style='text-align: center; color: #555;'>Record your voice using the recorder below and the transcript will appear.</p>", unsafe_allow_html=True)
    st.write("") # Spacer

    audio_processor = get_audio_processor()

    # Session state initialization
    if 'transcript_data' not in st.session_state:
        st.session_state.transcript_data = []
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = True

    # --- Recorder UI in a styled card ---
    st.markdown('<div class="recorder-card">', unsafe_allow_html=True)
    st.subheader("Recorder")
    
    # Use st.audio_input to record audio from the user's microphone
    audio_bytes = st.audio_input(
        "Record your voice here. Click the microphone to start and stop.",
        key='recorder',
        on_change=on_new_recording
    )
    st.markdown('</div>', unsafe_allow_html=True)


    # Process audio if a new recording is available and not yet processed
    if audio_bytes and not st.session_state.audio_processed:
        # Let the user listen to their recording
        st.audio(audio_bytes, format='audio/wav')
        
        with st.spinner("Transcribing and analyzing audio... This might take a moment."):
            try:
                # Get the bytes from the UploadedFile object
                audio_data = audio_bytes.getvalue()
                
                # Write bytes to a temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    tmp_wav.write(audio_data)
                    tmp_wav_path = tmp_wav.name
                
                transcript = audio_processor.process_audio(tmp_wav_path)
                
                if transcript:
                    st.session_state.transcript_data = transcript
                else:
                    st.warning("⚠️ No speech was detected in the audio. Please try recording again, speaking clearly into your microphone.")
                    st.session_state.transcript_data = []

                os.unlink(tmp_wav_path)
                st.session_state.audio_processed = True # Mark as processed
                st.rerun() # Rerun to update the view with the new transcript

            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
                st.error(f"An error occurred during processing: {e}")
                st.session_state.audio_processed = True # Also mark as processed on error to avoid loops
    
    # Display the editable transcript if it exists
    if st.session_state.transcript_data:
        st.divider()
        st.header("Editable Transcript")

        speakers = sorted(list(set(seg['speaker'] for seg in st.session_state.transcript_data)))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        speaker_colors = {speaker: colors[i % len(colors)] for i, speaker in enumerate(speakers)}

        for i, segment in enumerate(st.session_state.transcript_data):
            speaker, start_time, end_time = segment['speaker'], segment['start'], segment['end']
            ts_start = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
            ts_end = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}"
            timestamp_str = f"{ts_start} → {ts_end}"
            color = speaker_colors.get(speaker, "#808080")

            with st.container():
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 6px solid {color}; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                    <p class="speaker-name" style="color: {color};">{speaker}</p>
                    <p class="timestamp">{timestamp_str}</p>
                </div>
                """, unsafe_allow_html=True)
                
                new_text = st.text_area(label=f"Segment {i+1}", value=segment['text'], key=f"text_{i}", label_visibility="collapsed", height=100)

                if new_text != st.session_state.transcript_data[i]['text']:
                    st.session_state.transcript_data[i]['text'] = new_text
                    st.toast(f"Segment {i+1} updated!", icon="✅")
        
        st.divider()
        st.download_button(
            label="Download Edited Transcript (JSON)",
            data=json.dumps(st.session_state.transcript_data, indent=2),
            file_name="edited_transcript.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()