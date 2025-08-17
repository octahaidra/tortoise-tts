''' RunPod handler for Tortoise TTS '''

import os
from src.tts import TortoiseTextToSpeech
import base64

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_cleanup
import runpod.logging as logging
from rp_schema import INPUT_VALIDATIONS

logging.basicConfig(level=logging.INFO)
# Initialize logging
logging.info("Setting up the Whisper model...")
MODEL = TortoiseTextToSpeech()
MODEL.setup(use_deepspeed=True, use_kv_cache=True)  # Using optimized settings
logging.info("Tortoise TTS model setup complete.")

def run(job):
    '''
    Run inference on the Tortoise TTS model.
    Returns base64 encoded WAV audio data.
    '''
    logging.info("Received job input: %s", job.get('id', 'unknown'))
    job_input = job['input']
    logging.info(f"Job input: {job_input}")
    
    # Input validation
    logging.info("Validating input parameters...")
    validated_input = validate(job_input, INPUT_VALIDATIONS)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    logging.info("Input validation passed.")
    # Set the voice
    try:
        MODEL.set_voice(job_input.get('voice', 'train_atkins'))
    except Exception as e:
        return {"error": f"Error loading voice: {str(e)}"}
    
    # Set the preset if provided
    if 'preset' in job_input:
        MODEL.set_preset(job_input.get('preset', 'fast'))
    
    # Generate audio using either preset or advanced parameters
    logging.info("Generating audio...")
    try:
        if job_input.get('use_tts_with_preset', True):
            # Using the simpler tts_with_preset interface
            wav_bytes = MODEL.generate_audio(
                text=job_input['text'],
                return_wav=True
            )
        else:
            # Using the advanced tts interface with all available parameters
            wav_bytes = MODEL.generate_audio_advanced(
                text=job_input['text'],
                num_autoregressive_samples=job_input.get('num_autoregressive_samples', 512),
                temperature=job_input.get('temperature', 0.8),
                length_penalty=job_input.get('length_penalty', 1.0),
                repetition_penalty=job_input.get('repetition_penalty', 2.0),
                top_p=job_input.get('top_p', 0.8),
                max_mel_tokens=job_input.get('max_mel_tokens', 500),
                cvvp_amount=job_input.get('cvvp_amount', 0.0),
                diffusion_iterations=job_input.get('diffusion_iterations', 100),
                cond_free=job_input.get('cond_free', True),
                cond_free_k=job_input.get('cond_free_k', 2.0),
                sample_batch_size=job_input.get('sample_batch_size', 1),
                diffusion_temperature=job_input.get('diffusion_temperature', 1.0),
                return_wav=True
            )
            
        # Encode the WAV bytes to base64 for JSON transport
        logging.info("Encoding audio to base64...")
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        response = {
            "audio_base64": audio_base64,
            "voice": job_input.get('voice', 'train_atkins'),
            "preset": job_input.get('preset', 'fast')
        }
        # Clean up input objects to free memory
        rp_cleanup.clean(['input_objects'])
        logging.info("Audio generation completed successfully.")
        # Return the response
        return response
        
    except Exception as e:
        return {"error": f"Error generating audio: {str(e)}"}


runpod.serverless.start({"handler": run})

