#!/usr/bin/env python3
import sys
import time
import numpy as np
import logging
import sounddevice as sd
from whisper_streaming.whisper_online import asr_factory, set_logging

logger = logging.getLogger(__name__)

class ASRApp:
    def __init__(self):
        # Variables to hold settings
        self.model_var = 'tiny'
        self.input_var = 'Microphone'
        self.language_var = 'zh'
        self.task_var = 'transcribe'
        self.log_level_var = 'DEBUG'

    def start_asr(self):
        # ASR settings based on user input
        args = self.create_args()

        # Set logging
        set_logging(args, logger)

        # Initialize ASR
        self.asr, self.online = asr_factory(args)  # 确保使用faster-whisper

        SAMPLING_RATE = 16000  # Whisper requires 16kHz audio
        CHUNK_DURATION = args.min_chunk_size  # Chunk size in seconds
        CHUNK_SIZE = int(SAMPLING_RATE * CHUNK_DURATION)

        # Warm up model with silence, ensure it's in float32
        self.asr.transcribe(np.zeros(SAMPLING_RATE, dtype=np.float32))

        # Start capturing audio
        self.stream = sd.InputStream(samplerate=SAMPLING_RATE, channels=1, blocksize=CHUNK_SIZE, dtype='float32')
        self.stream.start()
        self.process_audio()

    def process_audio(self):
        try:
            # Continuously process audio
            while True:
                audio_data, _ = self.stream.read(self.stream.blocksize)
                audio = np.squeeze(audio_data)
                self.online.insert_audio_chunk(audio)
                result = self.online.process_iter()

                if result[0] is not None:
                    recognized_text = result[2]

                    # Check if recognized_text ends with a punctuation mark, implying a sentence is complete
                    if recognized_text.endswith(('.', '?', '!')):
                        recognized_text += '\n'  # Add newline for a complete sentence

                    # Print the recognized text to console
                    print(recognized_text)

        except Exception as e:
            logger.error(f"Error: {e}")
            self.stop_asr()

    def stop_asr(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            print("ASR stopped.")

    def create_args(self):
        class Args:
            pass

        args = Args()
        args.min_chunk_size = 0.8
        args.model = self.model_var
        args.model_cache_dir = None
        args.model_dir = None
        args.lan = self.language_var
        args.task = self.task_var
        args.backend = "faster-whisper"  # 强制使用faster-whisper
        args.vac = False
        args.vad = True  # 启用VAD
        args.buffer_trimming = "segment"
        args.buffer_trimming_sec = 15
        args.log_level = self.log_level_var

        return args

if __name__ == "__main__":
    app = ASRApp()
    app.start_asr()
