import streamlit as st
import os
import wave
import tempfile
import logging
import json
from dotenv import load_dotenv
from audio_processor import AudioProcessor
from streamlit_mic_recorder import mic_recorder

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
                whisper_model_size="small",
                auth_token=hf_token,
                compute_type="float32" # Use "int8" for faster CPU performance
            )
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize AudioProcessor: {e}")
        st.error(f"Error initializing audio processor: {e}")
        st.stop()

# --- Main App Logic ---
def main():
    st.title("🎙️ Live Audio Transcription & Diarization")
    st.markdown("<p style='text-align: center; color: #555;'>Click the button, speak, and see your editable transcript appear below.</p>", unsafe_allow_html=True)
    st.write("") # Spacer

    audio_processor = get_audio_processor()

    # Session state initialization
    if 'transcript_data' not in st.session_state:
        st.session_state.transcript_data = []

    # --- Recorder UI in a styled card ---
    st.markdown('<div class="recorder-card">', unsafe_allow_html=True)
    st.subheader("Recorder")
    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key='recorder',
        use_container_width= True
    )
    st.markdown('</div>', unsafe_allow_html=True)


    if audio and audio['bytes']:
        with st.spinner("Transcribing and analyzing audio... This might take a moment."):
            try:
                # Save audio bytes to a temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    with wave.open(tmp_wav.name, 'wb') as wf:
                        wf.setnchannels(1); wf.setsampwidth(2)
                        wf.setframerate(audio['sample_rate'])
                        wf.writeframes(audio['bytes'])
                    
                    transcript = audio_processor.process_audio(tmp_wav.name)
                
                # --- NEW: Check for empty transcript and provide feedback ---
                if transcript:
                    st.session_state.transcript_data = transcript
                else:
                    st.warning("⚠️ No speech was detected in the audio. Please try recording again, speaking clearly into your microphone.")
                    st.session_state.transcript_data = [] # Clear any previous transcript

                os.unlink(tmp_wav.name)
                st.rerun()

            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
                st.error(f"An error occurred during processing: {e}")
    
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