from whisper.asr import load_model

def run_speech_recognition(audio_file, whisper_arch, language, cuda_available, progress_callback=None, status_callback=None):
    """
    Function to run speech recognition on an audio file with progress and status updates.
    
    Args:
        audio_file (str): Path to the audio file.
        whisper_arch (str): Whisper model architecture to use.
        language (str): Language for transcription.
        cuda_available (bool): Whether to use CUDA.
        progress_callback (function, optional): A function to call with progress updates.
            The function should accept a single argument, an integer between 0 and 100.
        status_callback (function, optional): A function to call with status updates.
            The function should accept a single string argument representing the current status.
    
    Returns:
        dict: Transcription result.
    """
    device = "cuda" if cuda_available else "cpu"

    # Notify about model download.
    if status_callback:
        status_callback("Model downloading...")

    # Load the model and notify progress.
    model = load_model(whisper_arch=whisper_arch, device=device, language=language, download_root="model")

    # Notify that model download is complete.
    if progress_callback:
        progress_callback(20)
    if status_callback:
        status_callback("Model downloaded. Starting transcription...")

    # Start transcription and update status.
    if status_callback:
        status_callback("Starting transcription...")

    # Transcribe the audio file with progress updates.
    def internal_progress_callback(progress):
        # Notify progress updates during transcription.
        if progress_callback:
            # Map progress to a range from 20% to 100%
            adjusted_progress = 20 + int(progress * 0.8)
            progress_callback(adjusted_progress)
        if status_callback:
            status_callback(f"Transcribing... {adjusted_progress}% complete")

    transcription_result = model.transcribe(
        audio=audio_file,
        batch_size=1,
        print_progress=True,
        progress_callback=internal_progress_callback
    )
    
    # Notify that transcription is complete.
    if progress_callback:
        progress_callback(100)
    if status_callback:
        status_callback("Transcription complete.")

    return transcription_result
