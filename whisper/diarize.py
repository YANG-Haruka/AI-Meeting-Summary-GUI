import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
from pathlib import Path
import json
import os
from typing import Optional, Union
import tempfile


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        cache_dir=None
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token, cache_dir=cache_dir)
        if self.model:
            try:
                self.model = self.model.to(device)
            except Exception as e:
                print("Move Model To Device Error: \n", str(e))
                pass

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None, progress_callback=None):
        # If audio is a string, it is a file path, use it directly
        if isinstance(audio, str):
            audio_file = audio
        else:
            # If it's a NumPy array, save it as a temporary file
            audio_file = self.save_audio_to_tempfile(audio)

        # Define a ProgressHook class with callback
        class ProgressHookWithCallback(ProgressHook):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback

            def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
                super().__call__(step_name, step_artifact, file, total, completed)
                if self.callback:
                    self.callback(step_name, step_artifact, file, total, completed)

        # Use ProgressHook to display progress, passing the callback
        if progress_callback:
            with ProgressHookWithCallback(progress_callback) as hook:
                segments = self.model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers, hook=hook)
        else:
            segments = self.model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)

        # Convert separated segments to DataFrame format
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True))
        diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
        diarize_df.rename(columns={2: "speaker"}, inplace=True)

        return diarize_df

    def save_audio_to_tempfile(self, audio):
        """Save audio data as a temporary file"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            audio_file = tmp_wav.name
            waveform = torch.from_numpy(audio[None, :])
            torch.save(waveform, tmp_wav)
        return audio_file
    
import numpy as np

def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False, progress_callback=None):
    transcript_segments = transcript_result["segments"]
    total_segments = len(transcript_segments)
    
    for i, seg in enumerate(transcript_segments):
        # Update overall progress
        if progress_callback:
            progress_callback((i / total_segments) * 100)

        # Assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])

        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        
        # Assign speaker to words
        if 'words' in seg:
            total_words = len(seg['words'])
            for j, word in enumerate(seg['words']):
                # Update progress for each word
                if progress_callback:
                    segment_progress = (j / total_words) * (100 / total_segments)
                    progress_callback(((i / total_segments) * 100) + segment_progress)

                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])

                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    if progress_callback:
        progress_callback(100)  # Ensure progress reaches 100% upon completion

    return transcript_result

def save_transcription_with_speakers(transcription_result, output_dir="temp/text", output_file="transcription_diarized.json"):
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define file path
    file_path = os.path.join(output_dir, output_file)

    # Save result as JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(transcription_result, f, ensure_ascii=False, indent=4)

    print(f"Transcription and speaker diarization saved to: {file_path}")

class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
