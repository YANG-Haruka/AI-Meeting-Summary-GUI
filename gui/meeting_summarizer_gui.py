import os
import shutil
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from gui.gui import Ui_MainWindow
import torch
import json

from .ffmpeg_audio_extractor import AudioExtractorThread
from .speech_recognition import SpeechRecognitionThread
from .diarization_thread import DiarizationThread
from .summary_thread import SummaryThread
from summary.ollama_bot import populate_sum_model


LANGUAGE_MAP = {
    "日本語": "ja",
    "中文": "zh",
    "English": "en"
}

class DragDropButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super(DragDropButton, self).__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            print(f"Dropped video file path: {file_path}")
            self.setText(file_path)

def save_transcription_with_speakers(transcription_result, output_dir="result/text", output_file="transcription_diarized.json"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(transcription_result, f, ensure_ascii=False, indent=4)
    print(f"Transcription and speaker information saved to: {file_path}")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # Replace the original button with DragDropButton
        self.video_path = DragDropButton(self.centralwidget)
        self.video_path.setGeometry(QtCore.QRect(390, 100, 500, 80))
        self.video_path.setObjectName("video_path")
        self.video_path.setText("Open video")

        # Connect buttons to methods
        self.video_path.clicked.connect(self.open_file_dialog)
        self.speech2text.clicked.connect(self.start_speech2text)
        self.stop.clicked.connect(self.stop_and_cleanup)
        self.text_summary.clicked.connect(self.start_generate_summary)
        self.prompt_edit.clicked.connect(self.open_prompt_folder)

        self.temp_dir = "temp"
        self.hf_token_flag = True
        self.cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {self.cuda_available}")

        # Populate the sum_model combobox with ollama models
        self.populate_sum_model()
        self.load_prompts()
        self.target_language.currentIndexChanged.connect(self.load_prompts)

    def load_prompts(self):
        # Get the target language from the target_language combo box
        target_language = self.target_language.currentText()
        language = LANGUAGE_MAP.get(target_language, "en")
        prompt_folder = os.path.join("prompt", language)

        # Clear existing items in the prompt combo box
        self.prompt.clear()

        # Check if the folder exists and read JSON files
        if os.path.exists(prompt_folder):
            json_files = [f for f in os.listdir(prompt_folder) if f.endswith('.json')]
            json_files_no_extension = [os.path.splitext(f)[0] for f in json_files]
            if json_files:
                self.prompt.addItems(json_files_no_extension)
            else:
                self.prompt.addItem("No prompts available")
        else:
            print(f"Prompt folder for {target_language} does not exist.")

    def open_prompt_folder(self):
        # Get the target language and selected prompt filename
        target_language = self.target_language.currentText()
        language = LANGUAGE_MAP.get(target_language, "en")
        selected_prompt = self.prompt.currentText()
        
        # Construct the folder path
        prompt_folder = os.path.join("prompt", language)

        if selected_prompt != "No prompts available" and os.path.exists(prompt_folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(prompt_folder))
        else:
            QMessageBox.warning(self, "Folder Not Found", "The prompt folder does not exist or no prompt is selected.")

    def populate_sum_model(self):
        try:
            model_names = populate_sum_model()
            if model_names:
                self.sum_model.addItems(model_names)
            else:
                self.sum_model.addItem("Check ollama models")
        except Exception as e:
            print(f"Error fetching ollama models: {e}")
            QMessageBox.warning(self, "Model Load Failed", "Unable to detect ollama models, please check installation.")

    def open_file_dialog(self):
        self.status.setText("Selected Video.")
        self.update_progress(0)
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.mpeg *.mpg *.3gp *.m4v)", options=options)
        if file_path:
            print(f"Selected video file path: {file_path}")
            self.video_path.setText(file_path)

    def generate_file_name(self):
        video_file = self.video_path.text()
        hf_token = self.tf_token.text()

        if not video_file or not os.path.exists(video_file):
            self.status.setText("Please select a valid video file!")
            return False 

        if not hf_token:
            self.tf_token.setText("No Hugging Face Token entered; skipping speaker diarization.")
            self.hf_token_flag = False

        root_video_file = self.video_path.text()
        video_name = os.path.splitext(os.path.basename(root_video_file))[0]

        self.output_dir = f"result/{video_name}"
        self.output_file = "transcription_diarized.json" if self.hf_token_flag else "transcription.json"
        return True

    def start_speech2text(self):
        video_file = self.video_path.text()

        if not self.generate_file_name():
            return
        self.progressBar.setValue(0)
        self.status.setText("Extracting audio...")

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        audio_file = os.path.join(self.temp_dir, "extracted_audio.wav")
        self.extract_audio_thread = AudioExtractorThread(video_file, audio_file)
        self.extract_audio_thread.progress_updated.connect(self.update_progress)
        self.extract_audio_thread.finished.connect(lambda: self.start_speech_recognition(audio_file))
        self.extract_audio_thread.start()

    def start_speech_recognition(self, audio_file):
        self.status.setText("Speech transcription...")
        self.progressBar.setValue(0)

        whisper_arch = self.au_model.currentText()
        language = self.source_language.currentText()

        self.speech_recognition_thread = SpeechRecognitionThread(audio_file, whisper_arch, language, self.cuda_available)
        self.speech_recognition_thread.progress_updated.connect(self.update_progress)
        self.speech_recognition_thread.recognition_complete.connect(self.on_recognition_complete)
        self.speech_recognition_thread.start()

    def on_recognition_complete(self, transcription_result):
        hf_token = self.tf_token.text()
        if not self.hf_token_flag:
            self.status.setText("Speech transcription complete.")
            self.save_transcription(transcription_result)
            self.update_progress(100)
            return

        self.status.setText("Speaker diarization...")
        self.progressBar.setValue(0)

        self.diarization_thread = DiarizationThread(self.speech_recognition_thread.audio_file, transcription_result, hf_token, self.cuda_available)
        self.diarization_thread.progress_updated.connect(self.update_progress)
        self.diarization_thread.status_updated.connect(self.update_status_label)
        self.diarization_thread.diarization_complete.connect(self.on_diarization_complete)
        self.diarization_thread.start()

    def on_diarization_complete(self, final_transcription):
        self.status.setText("Completed! Results saved.")
        self.save_transcription(final_transcription)
        self.update_progress(100)

    def update_progress(self, value):
        self.progressBar.setValue(value)

    def update_status_label(self, status_text):
        self.status.setText(status_text)

    def save_transcription(self, final_transcription):
        save_transcription_with_speakers(final_transcription, self.output_dir, self.output_file)
    
    def start_generate_summary(self):
        if not self.generate_file_name():
            return

        self.progressBar.setValue(0)
        transcription_file = os.path.join(self.output_dir, self.output_file)
        model = self.sum_model.currentText()
        language = LANGUAGE_MAP.get(self.target_language.currentText(), "en")
        selected_prompt = self.prompt.currentText()
        prompt_folder = os.path.join("prompt", language)
        prompt_path = os.path.join(prompt_folder, f"{selected_prompt}.json")
        output_file = os.path.join(self.output_dir, "meeting_summary.json")

        self.summary_thread = SummaryThread(transcription_file, model, language, prompt_path, output_file)
        self.summary_thread.progress_updated.connect(self.update_progress)
        self.summary_thread.status_updated.connect(self.update_status_label)
        self.summary_thread.start()
        
    def stop_and_cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Temporary folder {self.temp_dir} deleted.")
        else:
            print("Temporary folder does not exist.")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
