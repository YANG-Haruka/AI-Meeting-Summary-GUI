import subprocess
import re

def extract_audio_from_video(video_file, audio_file, progress_callback=None):
    """
    Extracts audio from a video file using ffmpeg and reports progress.
    
    Args:
        video_file (str): Path to the video file.
        audio_file (str): Output path for the extracted audio file.
        progress_callback (function, optional): A function to call with progress updates.
            The function should accept a single argument, an integer between 0 and 100.
    
    Returns:
        str: Path to the extracted audio file.
    """
    # Get the total duration of the video for progress calculation
    total_duration = get_video_duration(video_file)

    command = [
        'ffmpeg',
        '-i', video_file,
        '-q:a', '0',
        '-map', 'a',
        audio_file,
        '-y'
    ]

    process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)

    # Read ffmpeg output line by line to extract progress information
    for line in process.stderr:
        if "time=" in line:
            match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
            if match:
                hours, minutes, seconds = map(float, match.groups())
                current_time = hours * 3600 + minutes * 60 + seconds
                progress = int((current_time / total_duration) * 100)
                
                # Call the progress callback function if provided
                if progress_callback:
                    progress_callback(progress)

    process.wait()

    # Ensure 100% progress is reported when done
    if progress_callback:
        progress_callback(100)

    return audio_file

def get_video_duration(video_file):
    """
    Gets the duration of a video file in seconds using ffprobe.
    
    Args:
        video_file (str): Path to the video file.
    
    Returns:
        float: Duration of the video in seconds.
    """
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return float(result.stdout)
