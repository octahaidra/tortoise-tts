''' RunPod handler for Tortoise TTS '''

import os
import base64
import logging
import runpod
from runpod.serverless.utils.rp_validator import validate
from tts import TortoiseTextToSpeech
from rp_schema import INPUT_VALIDATIONS

logging.basicConfig(level=logging.INFO)

class Handler:
    def __init__(self):
        self.MODEL = TortoiseTextToSpeech()

    def setup(self):
        """Initialize the model with optimal settings"""
        logging.info("Setting up the Tortoise TTS model...")
        self.MODEL.setup(
            use_deepspeed=True,
            use_kv_cache=True,
            use_float16=True
        )
        logging.info("Tortoise TTS model setup complete.")

    def run(self, job):
        '''
        Run text-to-speech inference.
        Returns base64 encoded WAV audio data.
        '''
        logging.info("Received job: %s", job.get('id', 'unknown'))
        job_input = job['input']

        # Input validation
        logging.info("Validating input...")
        validated_input = validate(job_input, INPUT_VALIDATIONS)
        if 'errors' in validated_input:
            return {"error": validated_input['errors']}
        
        try:
            # Set up model configuration
            self.MODEL.setup(
                use_deepspeed=job_input.get('use_deepspeed', True),
                use_kv_cache=job_input.get('use_kv_cache', True),
                use_float16=job_input.get('use_float16', True)
            )

            # Set voice
            voice = job_input.get('voice', 'train_atkins')
            logging.info(f"Setting voice to: {voice}")
            self.MODEL.set_voice(voice)

            # Set preset if provided
            if 'preset' in job_input:
                preset = job_input['preset']
                logging.info(f"Setting preset to: {preset}")
                self.MODEL.set_preset(preset)

            # Determine if we should use basic or advanced parameters
            use_advanced = any(key in job_input for key in [
                'num_autoregressive_samples', 'temperature', 'length_penalty',
                'repetition_penalty', 'top_p', 'max_mel_tokens', 'cvvp_amount',
                'diffusion_iterations', 'cond_free', 'cond_free_k',
                'sample_batch_size', 'diffusion_temperature'
            ])

            logging.info("Generating audio...")
            if use_advanced:
                # Advanced parameter usage
                # self.MODEL.settings (
                #     num_autoregressive_samples=job_input.get('num_autoregressive_samples', 512),
                #     temperature=job_input.get('temperature', 0.8),
                #     length_penalty=job_input.get('length_penalty', 1.0),
                #     repetition_penalty=job_input.get('repetition_penalty', 2.0),
                #     top_p=job_input.get('top_p', 0.8),
                #     max_mel_tokens=job_input.get('max_mel_tokens', 500),
                #     cvvp_amount=job_input.get('cvvp_amount', 0.0),
                #     diffusion_iterations=job_input.get('diffusion_iterations', 100),
                #     cond_free=job_input.get('cond_free', True),
                #     cond_free_k=job_input.get('cond_free_k', 2.0),
                #     sample_batch_size=job_input.get('sample_batch_size', 1),
                #     diffusion_temperature=job_input.get('diffusion_temperature', 1.0)
                # )
                wav_bytes = self.MODEL.generate_audio(
                    text=job_input['text'],
                    return_wav=True
                )
            else:
                # Basic parameter usage
                wav_bytes = self.MODEL.generate_audio(
                    text=job_input['text'],
                    return_wav=True
                )

            # Encode the WAV bytes to base64
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            return {
                "text": job_input['text'],
                "audio_base64": audio_base64,
                "voice": voice,
                "preset": job_input.get('preset', 'fast')
            }

        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            return {"error": str(e)}



