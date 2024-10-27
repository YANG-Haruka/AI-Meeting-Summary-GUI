from whisper.asr import load_model

def run_speech_recognition(audio_file, whisper_arch, language, cuda_available):
    """
    Function to run speech recognition on an audio file.
    
    Args:
        audio_file (str): Path to the audio file.
        whisper_arch (str): Whisper model architecture to use.
        language (str): Language for transcription.
        cuda_available (bool): Whether to use CUDA.
    
    Returns:
        dict: Transcription result.
    """
    device = "cuda" if cuda_available else "cpu"
    model = load_model(whisper_arch=whisper_arch, device=device, language=language, download_root="model")

    # Transcribe the audio file
    transcription_result = model.transcribe(audio=audio_file, batch_size=1)
    return transcription_result
