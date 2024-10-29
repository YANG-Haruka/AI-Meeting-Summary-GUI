import json
import ollama
import os
from ollama._types import Options

LANGUAGE_MAP = {
    "日本語": "ja",
    "中文": "zh",
    "English": "en"
}

def load_config(config_filepath):
    """Read configuration from config.json, including language for summary generation"""
    if not os.path.exists(config_filepath):
        print(f"Config file not found: {config_filepath}")
        return None

    try:
        with open(config_filepath, "r", encoding="utf-8") as file:
            config = json.load(file)
            return config
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

def load_segments_from_json(filepath):
    """Load transcription segments from a local JSON file"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            segments = data.get("segments", [])
            return segments
    except Exception as e:
        print(f"Error reading transcription from JSON: {e}")
        return None

def load_prompt_from_json(prompt_path):
    """Load prompt data from a language-specific JSON file in the prompt/meeting folder"""
    if not os.path.exists(prompt_path):
        print(f"Prompt file not found: {prompt_path}")
        return None

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_data = json.load(file)
            return prompt_data
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

def summarize_meeting(segments, model, gpt_dict_raw_text, prompt_path):
    """Generate a meeting summary with action items, using the Ollama API"""
    # Load user and system prompt data based on language
    prompt_data = load_prompt_from_json(prompt_path)
    if not prompt_data:
        print("Failed to load prompt data.")
        return None

    meeting_text = "\n".join([seg.get("text", "") for seg in segments])

    # Combine system and user prompts into one message
    combined_prompt = (
        f"{meeting_text}\n"
        f"{gpt_dict_raw_text}\n"
        f"{prompt_data.get('system_prompt', '')}\n"
        f"{prompt_data.get('user_prompt', '')}\n"
    )
    with open("test.json", "w", encoding="utf-8") as file:
            json.dump(combined_prompt, file, ensure_ascii=False, indent=4)

    # Send the combined prompt to Ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": combined_prompt}],
        options= Options(
                        num_ctx=10240,
                        num_predict=-1)
    )

    if response.get('done', False):
        summary = response['message']['content']
        return summary
    else:
        print("Failed to generate summary.")
        return None

def save_summary_to_markdown(summary, output_filepath):
    """Save the generated meeting summary to a Markdown file"""
    md_content = "# Meeting Summary\n\n"
    md_content += summary

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Meeting summary saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving meeting summary: {e}")

def populate_sum_model():
    """Check local Ollama models and return a list of model names"""
    try:
        models = ollama.list()
        if models and 'models' in models:
            model_names = [model['name'] for model in models['models']]
            return model_names
        else:
            return None
    except Exception as e:
        print(f"Error fetching ollama models: {e}")
        return None

if __name__ == "__main__":

    # print(populate_sum_model())
    
    # Configuration and file paths
    config_file = "config.json"
    transcription_file = "result/2024-10-24 16-03-39/transcription.json"
    output_file = "result/2024-10-24 16-03-39/meeting_summary.md"

    # Load configuration, model, and language
    config = load_config(config_file)
    model = config.get("model", "llama3.1:8b") if config else "llama3.1:8b"
    language = config.get("language", "English") if config else "English"

    # Load transcription segments
    segments = load_segments_from_json(transcription_file)
    gpt_dict_raw_text = " "  # Customize based on meeting content
    
    if segments:
        meeting_summary = summarize_meeting(segments, model, gpt_dict_raw_text, language)
        save_summary_to_markdown(meeting_summary, output_file)
    else:
        print("Failed to load transcription segments or prompt data.")
