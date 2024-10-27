from whisper.diarize import DiarizationPipeline, assign_word_speakers

def run_diarization(audio_file, transcription_result, hf_token, cuda_available):
    """
    Function to perform diarization on an audio file.
    
    Args:
        audio_file (str): Path to the audio file.
        transcription_result (dict): The transcription result.
        hf_token (str): Hugging Face token for authentication.
        cuda_available (bool): Whether CUDA is available for GPU acceleration.
    
    Returns:
        dict: Final transcription with speaker information.
    """
    device = "cuda" if cuda_available else "cpu"
    pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device, cache_dir="model")

    # Run audio separation and update progress (simulating progress)
    diarize_df = pipeline(audio_file)

    # Assign speakers to transcription
    final_transcription = assign_word_speakers(diarize_df, transcription_result)
    
    return final_transcription

