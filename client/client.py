"""
RunPod Tortoise TTS Client implementation.
Provides a client interface for interacting with the RunPod Tortoise TTS service.
"""

import os
import json
import time
import base64
import requests
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

from .exceptions import (
    ConfigurationError,
    ServerError,
    GenerationError,
    TimeoutError
)

class TortoiseTTSClient:
    """
    Client for interacting with RunPod Tortoise TTS service.
    
    This client handles configuration, connection management, and transcription
    requests to the Tortoise TTS service running on RunPod.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        pod_id: Optional[str] = None,
        port: Optional[str] = None,
        env_file: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Initialize the Tortoise TTS client.

        Args:
            api_key: RunPod API key. If not provided, will look for RUNPOD_API_KEY in env
            pod_id: RunPod Pod ID. If not provided, will look for POD_ID in env
            port: Pod port number. If not provided, will look for PORT in env or default to 3000
            env_file: Path to .env file for configuration
            timeout: Default timeout for requests in seconds
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        self.pod_id = pod_id or os.getenv('POD_ID')
        self.port = port or os.getenv('PORT', '3000')
        self.timeout = timeout

        if not self.api_key or not self.pod_id:
            raise ConfigurationError(
                "API key and Pod ID must be provided either through parameters or environment variables"
            )

        self.endpoint_url = f"https://{self.pod_id}-{self.port}.proxy.runpod.net"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def is_active(self) -> bool:
        """
        Check if the RunPod service is active and responding.

        Returns:
            bool: True if service is active, False otherwise
        """
        try:
            response = requests.get(
                f"{self.endpoint_url}/health",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _save_audio_file(self, audio_base64: str, output_path: str) -> None:
        """
        Save a base64 encoded WAV audio to a file.

        Args:
            audio_base64: Base64 encoded WAV audio data
            output_path: Path where to save the WAV file

        Raises:
            GenerationError: If there's an error saving the audio file
        """
        try:
            audio_data = base64.b64decode(audio_base64)
            with open(output_path, 'wb') as audio_file:
                audio_file.write(audio_data)
        except Exception as e:
            raise GenerationError(f"Error saving audio file: {str(e)}")

    def _wait_for_job(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for job completion with timeout.

        Args:
            job_id: The ID of the job to wait for

        Returns:
            dict: The completed job data

        Raises:
            TimeoutError: If job doesn't complete within timeout
            ServerError: If job fails or server returns error
        """
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                response = requests.get(
                    f"{self.endpoint_url}/status/{job_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                status_data = response.json()

                if status_data.get("status") == "COMPLETED":
                    return status_data
                elif status_data.get("status") == "FAILED":
                    raise ServerError(f"Job failed: {json.dumps(status_data, indent=2)}")

                time.sleep(2)

            except requests.exceptions.RequestException as e:
                raise ServerError(f"Error checking job status: {str(e)}")

        raise TimeoutError(f"Job did not complete within {self.timeout} seconds")

    def _extract_audio(self, data: Dict[str, Any]) -> str:
        """
        Extract audio data from the TTS response.

        Args:
            data: Response data containing audio_base64

        Returns:
            str: Base64 encoded audio data

        Raises:
            GenerationError: If audio cannot be extracted from response
        """
        try:
            if not data or 'output' not in data or 'audio_base64' not in data['output']:
                raise GenerationError("Invalid response format")
            return data['output']['audio_base64']
        except Exception as e:
            raise GenerationError(f"Error extracting audio from response: {str(e)}")

    def generate_speech(
        self,
        text: str,
        output_path: Optional[str] = None,
        voice: str = "train_atkins",
        preset: str = "fast",
        use_basic: bool = True,
        **kwargs
    ) -> Union[str, None]:
        """
        Generate speech from text using Tortoise TTS service.

        Args:
            text: The text to convert to speech
            output_path: Optional path to save the generated WAV file
            voice: Voice to use (e.g., "tom", "emma", etc.)
            preset: Quality preset ("ultra_fast", "fast", "standard", "high_quality")
            use_basic: Whether to use basic or advanced parameters
            **kwargs: Advanced parameters for the TTS model:
                Model Optimization:
                - use_deepspeed (bool, default=True): Use DeepSpeed for faster inference
                - use_kv_cache (bool, default=True): Use KV caching for faster inference
                - use_float16 (bool, default=True): Run model in float16/half precision

                Generation Parameters:
                - num_autoregressive_samples (int, default=512)
                - temperature (float, default=0.8)
                - length_penalty (float, default=1.0)
                - repetition_penalty (float, default=2.0)
                - top_p (float, default=0.8)
                - max_mel_tokens (int, default=500)
                - cvvp_amount (float, default=0.0)
                - diffusion_iterations (int, default=100)
                - cond_free (bool, default=True)
                - cond_free_k (float, default=2.0)
                - sample_batch_size (int, default=1)
                - diffusion_temperature (float, default=1.0)

        Returns:
            str or None: Base64 encoded WAV data if output_path is None,
                        otherwise None (saves to file)

        Raises:
            ConfigurationError: If input configuration is invalid
            ServerError: If server communication fails
            GenerationError: If speech generation fails
            TimeoutError: If request times out
        """
        # Prepare the payload based on basic or advanced parameters
        payload = {
            "input": {
                "text": text,
                "voice": voice,
                "preset": preset,
            }
        }

        # Add model optimization parameters
        model_params = {
            "use_deepspeed": kwargs.get("use_deepspeed", True),
            "use_kv_cache": kwargs.get("use_kv_cache", True),
            "use_float16": kwargs.get("use_float16", True)
        }
        payload["input"].update(model_params)

        # Add advanced parameters if needed
        if not use_basic:
            advanced_params = {
                "num_autoregressive_samples": kwargs.get("num_autoregressive_samples", 512),
                "temperature": kwargs.get("temperature", 0.8),
                "length_penalty": kwargs.get("length_penalty", 1.0),
                "repetition_penalty": kwargs.get("repetition_penalty", 2.0),
                "top_p": kwargs.get("top_p", 0.8),
                "max_mel_tokens": kwargs.get("max_mel_tokens", 500),
                "cvvp_amount": kwargs.get("cvvp_amount", 0.0),
                "diffusion_iterations": kwargs.get("diffusion_iterations", 100),
                "cond_free": kwargs.get("cond_free", True),
                "cond_free_k": kwargs.get("cond_free_k", 2.0),
                "sample_batch_size": kwargs.get("sample_batch_size", 1),
                "diffusion_temperature": kwargs.get("diffusion_temperature", 1.0)
            }
            payload["input"].update(advanced_params)

        try:
            # Send TTS request
            response = requests.post(
                f"{self.endpoint_url}/run",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            if "id" not in data:
                raise ServerError("No job ID in response")

            # Wait for job completion and get result
            result = self._wait_for_job(data["id"])

            # Extract audio data
            audio_base64 = self._extract_audio(result)

            # Save to file if output path is provided
            if output_path:
                self._save_audio_file(audio_base64, output_path)

            # return the base64 encoded audio
            return audio_base64

        except requests.exceptions.RequestException as e:
            raise ServerError(f"Error communicating with server: {str(e)}")
        except Exception as e:
            raise GenerationError(f"Speech generation failed: {str(e)}")

