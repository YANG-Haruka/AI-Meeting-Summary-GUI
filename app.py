import gradio as gr
import shutil
import torch
import os
import json

from online.diarization_thread import run_diarization
from online.ffmpeg_audio_extractor import extract_audio_from_video
from online.speech_recognition import run_speech_recognition
from online.summary_thread import generate_summary
from summary.ollama_bot import populate_sum_model

LANGUAGE_MAP = {
    "日本語": "ja",
    "中文": "zh",
    "English": "en"
}

def save_transcription_with_speakers(transcription_result, output_dir="result/text", output_file="transcription_diarized.json"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(transcription_result, f, ensure_ascii=False, indent=4)
    print(f"Transcription and speaker information saved to: {file_path}")
    return file_path

def speech2text(video_file, hf_token, whisper_model_name, source_language):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    audio_file = os.path.join(temp_dir, "extracted_audio.wav")

    try:
        # Extract audio from video
        extract_audio_from_video(video_file, audio_file)

        # Run speech recognition
        transcription_result = run_speech_recognition(audio_file, whisper_model_name, LANGUAGE_MAP.get(source_language), torch.cuda.is_available())

        # Run diarization if needed
        if hf_token:
            transcription_result = run_diarization(audio_file, transcription_result, hf_token, torch.cuda.is_available())

        # Save the transcription result
        transcription_file = save_transcription_with_speakers(transcription_result, "result", "transcription.json")

        return "Transcription complete.", transcription_file
    except Exception as e:
        return f"Error: {e}", None
    finally:
        shutil.rmtree(temp_dir)

def text_summary(transcription_file, llm_model_name, target_language, selected_prompt):
    prompt_path = f"prompt/{LANGUAGE_MAP.get(target_language, 'en')}/{selected_prompt}.json"
    summary_file = "result/meeting_summary.json"

    try:
        # Generate summary
        summary_file = generate_summary(transcription_file, llm_model_name, target_language, prompt_path, summary_file)
        return "Summary generation complete.", summary_file
    except Exception as e:
        return f"Error: {e}", None

def load_prompts(target_language):
    """Load prompt names based on the selected language."""
    language = LANGUAGE_MAP.get(target_language, "en")
    prompt_folder = os.path.join("prompt", language)

    if os.path.exists(prompt_folder):
        json_files = [f for f in os.listdir(prompt_folder) if f.endswith('.json')]
        return [os.path.splitext(f)[0] for f in json_files] or ["No prompts available"]
    else:
        return ["No prompts available"]

if __name__ == "__main__":
    whisper_models = ["large-v2", "large-v1", "medium", "small", "base", "tiny"]
    ollama_models = populate_sum_model() or ["None"]

    with gr.Blocks() as iface:
        gr.Markdown("# Video Summarizer")
        video_input = gr.Video(label="Upload a video file")
        hf_token_input = gr.Textbox(label="Hugging Face Token (Optional)", placeholder="Enter token for speaker diarization")
        whisper_model_input = gr.Dropdown(choices=whisper_models, label="Select a Whisper model", value=whisper_models[0])
        source_language_input = gr.Dropdown(choices=["English", "日本語", "中文"], label="Source Language", value="English")

        transcription_status = gr.Textbox(label="Status", interactive=False)
        transcription_file = gr.File(label="Download Transcription")

        speech2text_button = gr.Button("Run Speech to Text")

        # Speech-to-text step
        speech2text_button.click(
            fn=speech2text,
            inputs=[video_input, hf_token_input, whisper_model_input, source_language_input],
            outputs=[transcription_status, transcription_file]
        )

        llm_model_input = gr.Dropdown(choices=ollama_models, label="Select a summarization model", value=ollama_models[0])
        target_language_input = gr.Dropdown(choices=["English", "日本語", "中文"], label="Target Language", value="English")
        prompt_name_input = gr.Dropdown(label="Select a Prompt", choices=load_prompts("English"))

        # Load prompts when target language changes
        target_language_input.change(
            fn=load_prompts,
            inputs=target_language_input,
            outputs=prompt_name_input
        )

        summary_status = gr.Textbox(label="Summary Status", interactive=False)
        summary_file = gr.File(label="Download Summary")
        text_summary_button = gr.Button("Generate Summary")

        # Summary step
        text_summary_button.click(
            fn=text_summary,
            inputs=[transcription_file, llm_model_input, target_language_input, prompt_name_input],
            outputs=[summary_status, summary_file]
        )

    iface.launch()
