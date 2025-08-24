"""
RunPod Whisper STT Client Package.
This package provides a client interface for interacting with the RunPod Whisper STT service
and managing RunPod pods.
"""

from .client import WhisperSTTClient
from .pod import RunPodManager
from .exceptions import (
    TTSError,
    ConfigurationError,
    ServerError,
    GenerationError,
    TimeoutError,
    PodManagementError
)

__version__ = '0.1.0'
__all__ = [
    'WhisperSTTClient',
    'RunPodManager',
    'TTSError',
    'ConfigurationError',
    'ServerError',
    'GenerationError',
    'TimeoutError',
    'PodManagementError'
]
