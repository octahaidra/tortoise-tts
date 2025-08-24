"""
Example script demonstrating the usage of TTS and pod management.
This script shows how to:
1. Check and manage pod status
2. Wait for pod to be ready
3. Generate and play speech
4. Handle errors gracefully
"""

import os
import sys
import time
import base64
import sounddevice as sd
import numpy as np
import wave
import io
from pathlib import Path
from dotenv import load_dotenv
from client import (
    TortoiseTTSClient, RunPodManager, 
    PodManagementError, ConfigurationError, GenerationError
)

def main():
    # Load environment variables
    load_dotenv()
    
    # Get pod ID from environment
    pod_id = os.getenv('POD_ID')
    if not pod_id:
        print("Error: POD_ID environment variable not set")
        sys.exit(1)

    try:
        # Initialize the pod manager
        pod_manager = RunPodManager()
        print(f"\nChecking pod {pod_id} status...")
        
        # Check current pod status
        current_status = pod_manager.get_pod_status(pod_id)
        print(f"Current pod status: {current_status}")

        # If pod is stopped, resume it
        if current_status == "STOPPED" or current_status == "EXITED":
            print("\nPod is stopped. Resuming...")
            pod_manager.resume_pod(pod_id, gpu_count=1)
            
            print("Waiting for pod to be ready...")
            if not pod_manager.wait_for_pod_status(pod_id, "RUNNING", timeout=300):
                print("Error: Pod did not reach RUNNING status within timeout")
                sys.exit(1)
            print("Pod is now running!")

        def play_audio_from_base64(audio_base64: str):
            """Play audio from base64 string"""
            try:
                # Decode base64 to bytes
                audio_data = base64.b64decode(audio_base64)
                
                # Use wave to read the WAV data
                with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    sample_rate = wav_file.getframerate()
                    
                    # Play the audio
                    print("\nPlaying audio...")
                    sd.play(audio_array, sample_rate)
                    sd.wait()  # Wait until audio is finished playing
            except Exception as e:
                print(f"Error playing audio: {str(e)}")

        # Initialize the TTS client
        client = TortoiseTTSClient(api_key=os.getenv('RUNPOD_API_KEY'), pod_id=pod_id)
        
        # Wait for service to be active
        print("\nChecking if service is ready...")
        max_attempts = 30
        for attempt in range(max_attempts):
            if client.is_active():
                break
            print(f"Service not ready, waiting... ({attempt + 1}/{max_attempts})")
            time.sleep(10)
        else:
            print("Error: Service is not responding after waiting")
            sys.exit(1)
        print("Service is ready!")

        # Example text to convert to speech
        text = "Hello! This is a test of the Tortoise Text to Speech system."
        
        print(f"\nGenerating speech for text:")
        print("-" * 50)
        print(text)
        print("-" * 50)
        
        try:
            # Generate speech with optimal settings
            audio_base64 = client.generate_speech(
                text=text,
                voice="tom",  # You can try different voices: emma, tom, geralt, etc.
                preset="fast",
                use_deepspeed=True,
                use_kv_cache=True,
                use_float16=True
            )
            
            # Save the audio to a file
            output_file = "output.wav"
            print(f"\nSaving audio to {output_file}")
            client._save_audio_file(audio_base64, output_file)
            
            # Play the generated audio
            play_audio_from_base64(audio_base64)
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            sys.exit(1)

        print("\nSpeech generation completed successfully!")
        
        # Optionally stop the pod when done
        should_stop = input("\nWould you like to stop the pod? (y/N): ").lower()
        if should_stop == 'y':
            print("Stopping pod...")
            pod_manager.stop_pod(pod_id)
            print("Pod stopped successfully!")

    except PodManagementError as e:
        print(f"Pod management error: {str(e)}")
        sys.exit(1)
    except ConfigurationError as e:
        print(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
