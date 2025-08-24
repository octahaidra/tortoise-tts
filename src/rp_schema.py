INPUT_VALIDATIONS = {
    # Required parameters
    'text': {
        'type': str,
        'required': True,
        'description': 'The input text to synthesize into speech'
    },
    
    # Basic configuration
    'voice': {
        'type': str,
        'required': False,
        'default': 'train_atkins',
        'enum': [
            'angie', 'freeman', 'myself', 'tom', 'train_grace',
            'applejack', 'geralt', 'pat', 'train_atkins', 'train_kennard',
            'cond_latent_example', 'halle', 'pat2', 'train_daws', 'train_lescault',
            'daniel', 'jlaw', 'rainbow', 'train_dotrice', 'train_mouse',
            'deniro', 'lj', 'snakes', 'train_dreams', 'weaver',
            'emma', 'mol', 'tim_reynolds', 'train_empire', 'william'
        ],
        'description': 'Voice to use for synthesis. One of the available preset voices.'
    },
    'preset': {
        'type': str,
        'required': False,
        'default': 'fast',
        'enum': ['ultra_fast', 'fast', 'standard', 'high_quality'],
        'description': 'Quality/speed trade-off preset. ultra_fast=lowest quality/fastest, high_quality=best quality/slowest'
    },
    
    # Model configuration
    'use_deepspeed': {
        'type': bool,
        'required': False,
        'default': True,
        'description': 'Whether to use DeepSpeed for faster inference (2x speedup)'
    },
    'use_kv_cache': {
        'type': bool,
        'required': False,
        'default': True,
        'description': 'Whether to use KV caching for faster inference'
    },
    'use_float16': {
        'type': bool,
        'required': False,
        'default': True,
        'description': 'Whether to run the model in float16/half precision'
    },
    
    # Generation parameters
    'k': {
        'type': int,
        'required': False,
        'default': 1,
        'description': 'Number of candidate samples to generate. Larger k = more diverse but slower'
    },
    'num_autoregressive_samples': {
        'type': int,
        'required': False,
        'default': 512,
        'description': 'Number of AR samples. More samples = higher quality but slower'
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0.8,
        'description': 'Controls randomness in generation. Lower values = more deterministic'
    },
    
    # Advanced tuning
    'length_penalty': {
        'type': float,
        'required': False,
        'default': 1.0,
        'description': 'Penalty factor for sequence length. Higher values favor shorter sequences'
    },
    'repetition_penalty': {
        'type': float,
        'required': False,
        'default': 2.0,
        'description': 'Penalty factor for token repetition. Higher values prevent repeated words'
    },
    'top_p': {
        'type': float,
        'required': False,
        'default': 0.8,
        'description': 'Nucleus sampling cutoff. Smaller values = less diversity'
    },
    'max_mel_tokens': {
        'type': int,
        'required': False,
        'default': 500,
        'description': 'Maximum number of mel spectrogram tokens to generate'
    },
    
    # Diffusion model parameters
    'cvvp_amount': {
        'type': float,
        'required': False,
        'default': 0.0,
        'description': 'Amount of CVVP model to use (0.0 = disabled). Values >0 encourage voice consistency'
    },
    'diffusion_iterations': {
        'type': int,
        'required': False,
        'default': 100,
        'description': 'Number of diffusion steps. More iterations = smoother audio but slower'
    },
    'diffusion_temperature': {
        'type': float,
        'required': False,
        'default': 1.0,
        'description': 'Temperature parameter for the diffusion model'
    },
    
    # Conditioning parameters
    'cond_free': {
        'type': bool,
        'required': False,
        'default': True,
        'description': 'Whether to use conditioning-free guidance'
    },
    'cond_free_k': {
        'type': float,
        'required': False,
        'default': 2.0,
        'description': 'Conditioning-free guidance strength'
    },
    
    # Performance parameters
    'sample_batch_size': {
        'type': int,
        'required': False,
        'default': 1,
        'description': 'Batch size for parallel sample generation'
    },
    'verbose': {
        'type': bool,
        'required': False,
        'default': False,
        'description': 'Whether to log extra information during inference'
    }
}
}
