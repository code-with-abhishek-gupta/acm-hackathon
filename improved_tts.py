import pyttsx3
import threading
import logging

logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            # Set voice properties
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def speak(self, text: str, async_mode: bool = True):
        """Speak the given text"""
        if not self.engine:
            logger.warning("TTS engine not available")
            return
        
        try:
            if async_mode:
                # Run TTS in a separate thread to avoid blocking
                thread = threading.Thread(target=self._speak_sync, args=(text,))
                thread.daemon = True
                thread.start()
            else:
                self._speak_sync(text)
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
    
    def _speak_sync(self, text: str):
        """Synchronous speech method"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in synchronous TTS: {e}")

# Global TTS manager instance
tts_manager = TTSManager()

def speak_message(message: str):
    """Simple function to speak a message"""
    tts_manager.speak(message)
