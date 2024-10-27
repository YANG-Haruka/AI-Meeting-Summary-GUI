from summary.ollama_bot import load_segments_from_json,summarize_meeting,save_summary_to_json
from PyQt5 import QtCore

class SummaryThread(QtCore.QThread):
    progress_updated = QtCore.pyqtSignal(int)
    status_updated = QtCore.pyqtSignal(str)

    def __init__(self, transcription_file, model, language, prompt_path, output_file):
        super().__init__()
        self.transcription_file = transcription_file
        self.model = model
        self.language = language
        self.prompt_path = prompt_path
        self.output_file = output_file

    def run(self):
        try:
            segments = load_segments_from_json(self.transcription_file)
            if not segments:
                self.status_updated.emit("Failed to load transcription result.")
                return

            self.status_updated.emit("Generating meeting summary...")
            meeting_summary = summarize_meeting(segments, self.model, " ", self.prompt_path)
            save_summary_to_json(meeting_summary, self.output_file)

            self.status_updated.emit("Summary generated and saved.")
            self.progress_updated.emit(100)
        except Exception as e:
            print(f"Error generating meeting summary: {e}")
            self.status_updated.emit("Error generating meeting summary.")
