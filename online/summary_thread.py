from summary.ollama_bot import load_segments_from_json, summarize_meeting, save_summary_to_json

def generate_summary(transcription_file, model, language, prompt_path, output_file):
    """
    Function to generate a summary from the transcription.
    
    Args:
        transcription_file (str): Path to the transcription file.
        model (str): Model name for summarization.
        language (str): Language for summarization.
        prompt_path (str): Path to the prompt file.
        output_file (str): Path to save the summary.
    
    Returns:
        str: Path to the summary file.
    """
    segments = load_segments_from_json(transcription_file)
    if not segments:
        raise ValueError("Failed to load transcription result.")

    # Generate the summary
    meeting_summary = summarize_meeting(segments, model, " ", prompt_path)
    save_summary_to_json(meeting_summary, output_file)
    
    return output_file
