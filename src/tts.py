from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import io

class TortoiseTextToSpeech:
    """A class to handle text-to-speech conversion using Tortoise TTS."""
    
    def __init__(self):
        """Initialize the TTS class."""
        self.tts = None
        self.preset = "fast"  # default preset
        self.voice_samples = None
        self.conditioning_latents = None
        
    def setup(self, use_deepspeed=True, use_kv_cache=True):
        """
        Set up the TTS model and configure it.
        
        Args:
            use_deepspeed (bool): Whether to use deepspeed for faster inference
            use_kv_cache (bool): Whether to use KV caching for faster inference
        """
        self.tts = TextToSpeech(use_deepspeed=use_deepspeed, kv_cache=use_kv_cache)
        
    def set_voice(self, voice_name):
        """
        Set the voice to be used for synthesis.
        
        Args:
            voice_name (str): Name of the voice to use (e.g., 'tom', 'emma', etc.)
        """
        self.voice_samples, self.conditioning_latents = load_voice(voice_name)
        
    def set_preset(self, preset):
        """
        Set the quality preset for synthesis.
        
        Args:
            preset (str): One of {"ultra_fast", "fast", "standard", "high_quality"}
        """
        if preset not in ["ultra_fast", "fast", "standard", "high_quality"]:
            raise ValueError("Invalid preset. Must be one of: ultra_fast, fast, standard, high_quality")
        self.preset = preset
        
    def generate_audio(self, text, return_wav=True):
        """
        Generate audio from text using the configured settings.
        
        Args:
            text (str): The text to convert to speech
            return_wav (bool): If True, returns WAV file bytes, if False returns tensor
            
        Returns:
            bytes or torch.Tensor: WAV file bytes if return_wav=True, otherwise audio tensor
        """
        if self.tts is None:
            raise RuntimeError("TTS model not initialized. Call setup() first.")
        if self.voice_samples is None:
            raise RuntimeError("Voice not set. Call set_voice() first.")
            
        gen = self.tts.tts_with_preset(
            text,
            voice_samples=self.voice_samples,
            conditioning_latents=self.conditioning_latents,
            preset=self.preset
        )
        
        if return_wav:
            wav_buffer = io.BytesIO()
            torchaudio.save(wav_buffer, gen.squeeze(0).cpu(), 24000, format="wav")
            return wav_buffer.getvalue()
            
        return gen


def __main__():
    """
    Example usage of the TortoiseTextToSpeech class.
    This function can be used to test the TTS functionality.
    """
    import IPython
    tts = TortoiseTextToSpeech()
    tts.setup()
    # Choose a voice (str): one of voices = 
    """ {"angie", "freeman", "myself", "tom", "train_grace",
        "applejack", "geralt", "pat", "train_atkins", "train_kennard",
        "cond_latent_example", "halle", "pat2", "train_daws", "train_lescault",
        "daniel", "jlaw", "rainbow", "train_dotrice", "train_mouse",
        "deniro", "lj", "snakes", "train_dreams", "weaver",
        "emma", "mol", "tim_reynolds", "train_empire", "william"
    }"""
    tts.set_voice('tom') # Choose a voice
    tts.set_preset('fast') #preset (str): One of {"ultra_fast", "fast", "standard", "high_quality"}
    output_binary = tts.generate_audio("This is a test of the Tortoise text to speech system.")
    IPython.display.Audio(output_binary)
    


