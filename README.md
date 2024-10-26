# AI-Meeting-Summary-GUI

You need to install CUDA and cuDNN
(Currently there are no problems with 11.7 and 12.1 tests)

You need to download ollama and download the model for summarization
Download Ollama 
https://ollama.com/
Download model(For example llama3.1) 
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