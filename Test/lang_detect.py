import speech_recognition as sr
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pyaudio

def detect_language_from_speech():
    # Initialize recognizer and microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Adjust for ambient noise
    print("Adjusting for ambient noise... Please wait.")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    
    print("Listening... Speak now!")
    
    # Record audio for exactly 10 seconds
    try:
        with microphone as source:
            print("Recording for 10 seconds... Start speaking now!")
            # Record for exactly 10 seconds
            audio = recognizer.record(source, duration=10)
        print("Recording complete. Processing...")
    except sr.WaitTimeoutError:
        print("No speech detected within timeout period.")
        return
    
    # Define languages to test
    languages = ["en-US", "hi-IN", "es-ES", "fr-FR", "de-DE"]  # English, Hindi, Spanish, French, German
    results = {}
    
    # Test each language
    for lang in languages:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text.strip():  # Only add non-empty results
                results[lang] = text
                print(f"[{lang}] Recognition: {text}")
        except sr.UnknownValueError:
            # Speech was unintelligible for this language
            continue
        except sr.RequestError as e:
            print(f"Error with {lang}: {e}")
            continue
    
    if not results:
        print("No speech could be recognized in any of the tested languages.")
        return
    
    # Method 1: Pick the longest transcription (usually more accurate)
    best_lang_by_length = max(results, key=lambda k: len(results[k]))
    print(f"\nBest by length - Language: {best_lang_by_length}, Text: '{results[best_lang_by_length]}'")
    
    # Method 2: Use langdetect on the transcribed text
    try:
        # Use the longest transcription for language detection
        detected_lang = detect(results[best_lang_by_length])
        print(f"Langdetect result: {detected_lang}")
        
        # Map langdetect codes to our language codes
        lang_mapping = {
            'en': 'en-US',
            'hi': 'hi-IN', 
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE'
        }
        
        mapped_lang = lang_mapping.get(detected_lang, detected_lang)
        if mapped_lang in results:
            print(f"Final detected language: {mapped_lang}")
            print(f"Final transcription: '{results[mapped_lang]}'")
        else:
            print(f"Langdetect suggests {detected_lang}, but we don't have a transcription for that.")
            print(f"Using best by length: {best_lang_by_length}")
            
    except LangDetectException:
        print("Could not detect language using langdetect. Using longest transcription.")
        print(f"Final result - Language: {best_lang_by_length}, Text: '{results[best_lang_by_length]}'")
    
    return results

def test_with_audio_file(audio_file_path):
    """Alternative method to test with an audio file instead of microphone"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return
    
    languages = ["en-US", "hi-IN", "es-ES", "fr-FR", "de-DE"]
    results = {}
    
    for lang in languages:
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text.strip():
                results[lang] = text
                print(f"[{lang}] Recognition: {text}")
        except:
            continue
    
    if results:
        best_lang = max(results, key=lambda k: len(results[k]))
        print(f"\nBest result - Language: {best_lang}, Text: '{results[best_lang]}'")
    else:
        print("No speech recognized in any language.")

if __name__ == "__main__":
    # Test with microphone
    # detect_language_from_speech()
    
    # Uncomment below to test with an audio file instead
    test_with_audio_file("C:\\Users\\Admin\\Desktop\\Voice Demo\\Test\\my_voice_recording.wav")