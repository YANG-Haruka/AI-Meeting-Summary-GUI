# AI-Meeting-Summary-GUI

You need to install CUDA and cuDNN
(Currently there are no problems with 11.7 and 12.1 tests)  

To extract audio, you need to download ffmpeg  
https://www.ffmpeg.org/

You need to download ollama and download the model for summarization  

Download Ollama  
https://ollama.com/  

Download model (For example llama3.1) 
```bash
ollama pull llama3.1
```

Install torch
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install other requirements
```bash
pip install -r requirements.txt
```

Run the tool
```bash
python main.py
```


"Click or drag a video into the 'Open video' button to select a video.  

You can choose a speech recognition model and a summarization model. The summarization model is downloaded from Ollama.  

The source language is the language of the video, while the target language is the language in which the summarization results will be displayed.  

The right-side prompt includes some built-in prompts. Please select one according to your needs, or you can customize it.  

Click 'Speech-to-Text' to generate transcription results. (The first time requires downloading the Whisper model, which may take some time. To check the download progress, please view the terminal.)  

Click 'Text Summary' to summarize the transcription results using the selected summarization model.  

Audio results are saved in the ```temp``` folder.  
Transcription and summary results are saved in the ```result/{video_name}``` folder.  
Click 'Stop' to clear the temp folder."  
