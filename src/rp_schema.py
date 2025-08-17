INPUT_VALIDATIONS = {
    'text': {
        'type': str,
        'required': True,
        'description': 'The input text to synthesize'
    },
    'voice': {
        'type': str,
        'required': False,
        'default': 'train_atkins',
        'description': 'Voice name (e.g., "train_atkins", "lj", "emma")'
    },
    'preset': {
        'type': str,
        'required': False,
        'default': 'fast',
        'enum': ['ultra_fast', 'fast', 'standard', 'high_quality'],
        'description': 'Quality/speed trade-off preset'
    },
    'k': {
        'type': int,
        'required': False,
        'default': 1,
        'description': 'Number of candidate samples to generate'
    },
    'verbose': {
        'type': bool,
        'required': False,
        'default': False,
        'description': 'Whether to log extra info during inference'
    },
    # Advanced parameters
    'num_autoregressive_samples': {
        'type': int,
        'required': False,
        'default': 512,
        'description': 'Number of AR samples for higher quality'
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0.8,
        'description': 'Controls randomness in generation'
    },
    'length_penalty': {
        'type': float,
        'required': False,
        'default': 1.0,
        'description': 'Prevents word repetition'
    },
    'repetition_penalty': {
        'type': float,
        'required': False,
        'default': 2.0,
        'description': 'Prevents word repetition'
    },
    'top_p': {
        'type': float,
        'required': False,
        'default': 0.8,
        'description': 'Nucleus sampling cutoff'
    },
    'max_mel_tokens': {
        'type': int,
        'required': False,
        'default': 500,
        'description': 'Maximum number of mel tokens'
    },
    'cvvp_amount': {
        'type': float,
        'required': False,
        'default': 0.0,
        'description': 'CVVP model usage amount'
    },
    'diffusion_iterations': {
        'type': int,
        'required': False,
        'default': 100,
        'description': 'Number of diffusion iterations'
    },
    'cond_free': {
        'type': bool,
        'required': False,
        'default': True,
        'description': 'Use conditioning-free guidance'
    },
    'cond_free_k': {
        'type': float,
        'required': False,
        'default': 2.0,
        'description': 'Conditioning-free guidance amount'
    },
    'sample_batch_size': {
        'type': int,
        'required': False,
        'default': 1,
        'description': 'Batch size for sampling'
    },
    'diffusion_temperature': {
        'type': float,
        'required': False,
        'default': 1.0,
        'description': 'Temperature for diffusion model'
    }
}
