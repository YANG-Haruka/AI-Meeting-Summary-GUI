from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
from whisper.diarize import DiarizationPipeline, assign_word_speakers

class DiarizationThread(QThread):
    progress_updated = pyqtSignal(int)  # Signal to update the progress bar
    status_updated = pyqtSignal(str)  # Signal to update status description
    diarization_complete = pyqtSignal(dict)  # Signal to return final result (includes transcription and speaker separation)

    def __init__(self, audio_file, transcription_result, hf_token, cuda_available):
        super().__init__()
        self.audio_file = audio_file
        self.transcription_result = transcription_result
        self.hf_token = hf_token
        self.device = "cuda" if cuda_available else "cpu"

    def run(self):
        pipeline = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device, cache_dir="model")

        def update_progress(step_name, step_artifact, file=None, total=None, completed=None):
            if total is None or completed is None:
                total = 1
                completed = 1
            progress = (completed / total) * 100
            self.progress_updated.emit(int(progress))
            self.status_updated.emit(f"Separation - {step_name}")
        
        def progress_callback(progress):
            self.progress_updated.emit(int(progress))

        # Step 1: Run audio separation and update progress
        self.status_updated.emit("Audio separation...")
        diarize_df = pipeline(self.audio_file, progress_callback=update_progress)

        # Step 2: Separation complete, update progress bar to 50%, prepare to assign speakers
        self.status_updated.emit("Separation complete, assigning speakers...")
        self.progress_updated.emit(0)
        final_transcription = assign_word_speakers(diarize_df, self.transcription_result, progress_callback=progress_callback)

        # Step 3: Speaker assignment complete, update progress bar to 100%
        self.status_updated.emit("Speaker assignment complete.")
        self.progress_updated.emit(100)

        # Emit final result containing transcription and speaker information
        self.diarization_complete.emit(final_transcription)
