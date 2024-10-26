from PyQt5.QtCore import QThread, pyqtSignal
import torch
from whisper.asr import load_model

class SpeechRecognitionThread(QThread):
    progress_updated = pyqtSignal(int)  # Signal to update the progress bar
    recognition_complete = pyqtSignal(dict)  # Signal to indicate recognition completion

    LANGUAGE_MAP = {
        "日本語": "ja",
        "中文": "zh",
        "English": "en"
    }

    def __init__(self, audio_file, whisper_arch, language, cuda_available):
        super().__init__()
        self.audio_file = audio_file
        self.whisper_arch = whisper_arch
        self.language = self.LANGUAGE_MAP.get(language, "en")
        self.device = "cuda" if cuda_available else "cpu"

    def run(self):
        """
        Run the speech recognition model and emit progress.
        """
        model = load_model(
            whisper_arch=self.whisper_arch,
            device=self.device,
            language=self.language,
            download_root="model"
        )

        def progress_callback(progress):
            print(f"Progress: {progress}%")  # Debugging print
            self.progress_updated.emit(int(progress))

        # Transcribe audio and track progress
        transcription_result = model.transcribe(
            audio=self.audio_file,
            batch_size=1,
            print_progress=True,
            progress_callback=progress_callback
        )

        # Emit final result upon completion
        self.recognition_complete.emit(transcription_result)