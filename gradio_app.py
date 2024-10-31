import gradio as gr
import shutil
import torch
import os
import json

from gr_processing.ffmpeg_audio_extractor import extract_audio_from_video
from gr_processing.speech_recognition import run_speech_recognition
from gr_processing.summary_thread import generate_summary
from summary.ollama_bot import populate_sum_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

LANGUAGE_MAP = {
    "日本語": "ja",
    "中文": "zh",
    "English": "en"
}

def save_transcription_with_speakers(transcription_result, video_name, output_file="transcription.json"):
    output_dir = os.path.join("result", video_name)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(transcription_result, f, ensure_ascii=False, indent=4)
    return file_path

def speech2text(video_file, whisper_model_name, source_language, progress=gr.Progress()):
    if not video_file:
        return "No video selected", None

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    audio_file = os.path.join(temp_dir, "extracted_audio.wav")

    status_message = "Starting transcription..."
    transcription_file = None

    try:
        def status_callback(status):
            nonlocal status_message, current_progress
            status_message = status
            progress(current_progress/100, status_message)

        def transcription_progress_callback(p):
            nonlocal current_progress
            current_progress = p
            progress(current_progress, f"Transcribing audio: {int(p * 100)}%")
        
        current_progress = 0
        # Update status to indicate audio extraction
        status_callback("Extracting audio from video...")
        extract_audio_from_video(
            video_file,
            audio_file,
            progress_callback=lambda p: progress((p / 100), f"Extracting audio: {p}%")
        )

        # Update status to indicate speech recognition
        status_callback("Running speech recognition...")
        
        transcription_result = run_speech_recognition(
            audio_file,
            whisper_model_name,
            LANGUAGE_MAP.get(source_language),
            torch.cuda.is_available(),
            progress_callback=transcription_progress_callback,
            status_callback=status_callback
        )

        # Update status to indicate saving the transcription result
        status_callback("Saving transcription result...")
        transcription_file = save_transcription_with_speakers(
            transcription_result, video_name, "transcription.json"
        )

        current_progress = 100
        status_callback("Transcription complete.")
    except Exception as e:
        status_message = f"Error: {e}"
        progress(0, status_message)
    finally:
        shutil.rmtree(temp_dir)

    return status_message, transcription_file

def text_summary(llm_model_name, target_language, selected_prompt, video_file):
    if not video_file:
        return "No video selected", None

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    prompt_path = f"prompt/{LANGUAGE_MAP.get(target_language, 'en')}/{selected_prompt}.json"
    summary_file = os.path.join("result", video_name, "meeting_summary.md")
    transcription_file = os.path.join("result", video_name, "transcription.json")

    try:
        # Generate summary
        summary_file = generate_summary(transcription_file, llm_model_name, target_language, prompt_path, summary_file)
        return "Summary generation complete.", summary_file
    except Exception as e:
        return f"Error: {e}", None

def load_prompts(target_language):
    """Load prompt names based on the selected language, adding default prompts if available."""
    language = LANGUAGE_MAP.get(target_language, "en")
    prompt_folder = os.path.join("prompt", language)

    # Define default prompt filenames based on language
    default_prompts = {
        "en": "Default-Meeting Summary.json",
        "ja": "Default-会議の要約.json",
        "zh": "默认-会议总结.json"
    }

    prompts = []
    if os.path.exists(prompt_folder):
        # Load all available prompts in the folder
        json_files = [f for f in os.listdir(prompt_folder) if f.endswith('.json')]

        # Add default prompt first if it exists
        default_prompt = default_prompts.get(language)
        if default_prompt in json_files:
            prompts.append(os.path.splitext(default_prompt)[0])

        # Add the rest of the available prompts, excluding duplicates
        prompts.extend([os.path.splitext(f)[0] for f in json_files if f != default_prompt])

    return prompts or ["No prompts available"]

def save_prompt(language, prompt_name, prompt_content):
    """Save a new prompt or edit an existing one."""
    language_folder = os.path.join("prompt", LANGUAGE_MAP.get(language, "en"))
    os.makedirs(language_folder, exist_ok=True)
    prompt_path = os.path.join(language_folder, f"{prompt_name}.json")

    try:
        # Parse JSON content to validate it
        prompt_data = json.loads(prompt_content)
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=4)
        return f"Prompt '{prompt_name}' saved successfully."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. {e}"

def load_prompt_content(language, prompt_name):
    """Load the content of the selected prompt for editing."""
    prompt_path = os.path.join("prompt", LANGUAGE_MAP.get(language, "en"), f"{prompt_name}.json")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f), indent=4)
    else:
        return ""

if __name__ == "__main__":
    whisper_models = ["large-v2", "large-v1", "medium", "small", "base", "tiny"]
    ollama_models = populate_sum_model() or ["None"]

    with gr.Blocks() as iface:
        gr.Markdown("# Video Summarizer")
        with gr.Row():
            with gr.Column():
                # Speech to Text Module
                gr.Markdown("## Speech to Text")
                video_input = gr.Video(
                    label="Upload a video file",
                    format=None
                )
                whisper_model_input = gr.Dropdown(choices=whisper_models, label="Select a Whisper model", value=whisper_models[0])
                source_language_input = gr.Dropdown(choices=["English", "日本語", "中文"], label="Source Language", value="English")

                transcription_status = gr.Textbox(label="Status", interactive=False)
                transcription_file = gr.File(label="Download Transcription")

                speech2text_button = gr.Button("Run Speech to Text")

                # Speech-to-text step
                speech2text_button.click(
                    fn=speech2text,
                    inputs=[video_input, whisper_model_input, source_language_input],
                    outputs=[transcription_status, transcription_file]
                )

            with gr.Column():
                # Generate Summary Module
                gr.Markdown("## Generate Summary")
                llm_model_input = gr.Dropdown(choices=ollama_models, label="Select a summarization model", value=ollama_models[0])
                target_language_input = gr.Dropdown(choices=["English", "日本語", "中文"], label="Target Language", value="English")
                
                # Initialize prompt_name_input with no choices, to be loaded dynamically
                prompt_name_input = gr.Dropdown(label="Select a Prompt", choices=[])

                # Load prompts when target language changes and on initial load
                def update_prompts(language):
                    prompts = load_prompts(language)
                    default_value = prompts[0] if prompts else None
                    return gr.update(choices=prompts, value=default_value)

                # Update prompt_name_input when the target language changes
                target_language_input.change(
                    fn=update_prompts,
                    inputs=[target_language_input],
                    outputs=prompt_name_input
                )

                iface.load(
                    fn=update_prompts,
                    inputs=[target_language_input],
                    outputs=prompt_name_input
                )

                # Buttons for creating, editing, and deleting prompts
                with gr.Row():
                    new_prompt_button = gr.Button("New Prompt")
                    edit_prompt_button = gr.Button("Edit Selected Prompt")
                    delete_prompt_button = gr.Button("Delete Selected Prompt")

                # New/Edit Prompt Textbox and Save Button
                prompt_content_input = gr.Textbox(label="Prompt Content", placeholder="Enter JSON content for the prompt", visible=False)
                prompt_name_input_for_edit = gr.Textbox(label="Prompt Name", placeholder="Enter the name for the prompt", visible=False)
                save_prompt_button = gr.Button("Save Prompt", visible=False)

                summary_status = gr.Textbox(label="Summary Status", interactive=False)
                summary_file = gr.File(label="Download Summary")

                # Handle creating a new prompt
                def create_new_prompt(language):
                    # Set default values for new prompt creation
                    prompt_name = "User_Prompt"
                    prompt_content = json.dumps({
                        "system_prompt": "",
                        "user_prompt": ""
                    }, indent=4)
                    return (
                        prompt_content,
                        gr.update(value=prompt_name, visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True)
                    )

                new_prompt_button.click(
                    fn=create_new_prompt,
                    inputs=[target_language_input],
                    outputs=[prompt_content_input, prompt_name_input_for_edit, save_prompt_button, prompt_content_input]
                )

                # Handle editing an existing prompt
                edit_prompt_button.click(
                    fn=lambda language, prompt_name: (
                        load_prompt_content(language, prompt_name),
                        gr.update(visible=True),
                        gr.update(value=prompt_name, visible=True),
                        gr.update(visible=True)
                    ),
                    inputs=[target_language_input, prompt_name_input],
                    outputs=[prompt_content_input, prompt_content_input, prompt_name_input_for_edit, save_prompt_button]
                )

                # Save the prompt and refresh the prompt list
                def save_and_update_prompts(language, prompt_name, content):
                    # Save the new or edited prompt
                    save_message = save_prompt(language, prompt_name, content)
                    # Get the updated list of prompts for the selected language
                    updated_prompts = load_prompts(language)
                    return save_message, gr.update(choices=updated_prompts, value=prompt_name)

                save_prompt_button.click(
                    fn=save_and_update_prompts,
                    inputs=[target_language_input, prompt_name_input_for_edit, prompt_content_input],
                    outputs=[summary_status, prompt_name_input]
                )

                # Handle deleting a selected prompt and refreshing the prompt list
                def delete_prompt(language, prompt_name):
                    language_folder = os.path.join("prompt", LANGUAGE_MAP.get(language, "en"))
                    prompt_path = os.path.join(language_folder, f"{prompt_name}.json")
                    if os.path.exists(prompt_path):
                        os.remove(prompt_path)
                        message = f"Prompt '{prompt_name}' deleted successfully."
                    else:
                        message = f"Prompt '{prompt_name}' does not exist."
                    
                    # Refresh the list of prompts
                    updated_prompts = load_prompts(language)
                    new_value = updated_prompts[0] if updated_prompts else None
                    return message, gr.update(choices=updated_prompts, value=new_value)

                delete_prompt_button.click(
                    fn=delete_prompt,
                    inputs=[target_language_input, prompt_name_input],
                    outputs=[summary_status, prompt_name_input]
                )

                # Summary step
                text_summary_button = gr.Button("Generate Summary")
                text_summary_button.click(
                    fn=text_summary,
                    inputs=[llm_model_input, target_language_input, prompt_name_input, video_input],
                    outputs=[summary_status, summary_file]
                )

    iface.launch()
    # iface.launch(share=True)
