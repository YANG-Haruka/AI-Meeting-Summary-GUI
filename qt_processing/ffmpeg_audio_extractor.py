import subprocess
import re
from PyQt5.QtCore import QThread, pyqtSignal

class AudioExtractorThread(QThread):
    """
    Thread to extract audio from a video file using ffmpeg and update progress.
    """
    progress_updated = pyqtSignal(int)

    def __init__(self, video_file, audio_file):
        super().__init__()
        self.video_file = video_file
        self.audio_file = audio_file

    def run(self):
        total_duration = self.get_video_duration(self.video_file)

        command = [
            'ffmpeg',
            '-i', self.video_file,
            '-q:a', '0',
            '-map', 'a',
            self.audio_file,
            '-y'
        ]

        process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)

        for line in process.stderr:
            if "time=" in line:
                match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
                if match:
                    hours, minutes, seconds = map(float, match.groups())
                    current_time = hours * 3600 + minutes * 60 + seconds
                    progress = int((current_time / total_duration) * 100)
                    self.progress_updated.emit(progress)

        process.wait()

    def get_video_duration(self, video_file):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        return float(result.stdout)
