"""
Custom exceptions for the Whisper STT client.
"""

class TTSError(Exception):
    """Base exception for Tortoise TTS client errors."""
    pass

class ConfigurationError(TTSError):
    """Raised when there's an error in the client configuration."""
    pass

class ServerError(TTSError):
    """Raised when there's an error communicating with the server."""
    pass

class GenerationError(TTSError):
    """Raised when there's an error during transcription."""
    pass

class TimeoutError(TTSError):
    """Raised when a request times out."""
    pass

class PodManagementError(TTSError):
    """Raised when there's an error managing RunPod pods."""
    pass
