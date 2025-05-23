# AI-dubbing is a semi-automatic video (podcasts, movies, series) Dubbing system.
## Translate any video into your native language and keep the voices of the characters.

The project uses the following AI models:

OpenAI Whisper, 
Pyannote Speaker Diarization, 
Facebookresearch demucs MDXnet, 
SWivid F5-TTS

It's a hobby project. I'll just a little maintan.

At the moment It run on linux only with anaconda it use 2 enviroment.

Prepare F5-TTS anaconda enviroment:
```
conda create -n f5-tts python=3.10 && conda activate f5-tts
conda install git
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/SWivid/F5-TTS.git
pip install num2words
conda deactivate
```

Prepare AI_dubbing anaconda enviroment:
```
conda  create -n sync python=3.10 && conda activate sync
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda install git
pip install git+https://github.com/m-bain/whisperx.git
conda install cudatoolkit=11.8 cudnn=8.*
git clone https://github.com/sarpba/AI_Dubbing.git
cd AI_Dubbing
pip install -r requirements.txt
sudo apt install ffmpeg
sudo apt install gcc
pip install webrtcvad
```
Edit the whisperx alignment.py, and change the default aligment models to better if have. It's important for more precise alignment process.

Copy your F5-TTS model.pt and vocab.txt into the TTS directory. 

Create a Hugging Face account (it's optional, only need if you want to use speaker diarization), and accept the licenses for the following:
Pyannote Speaker Diarization 3.1 Generate a read API key and save it securely. https://huggingface.co/pyannote/speaker-diarization-3.1


Register for a DeepL account, activate the API with a free subscription, and generate an API key as well. This free tier includes up to 500,000 characters per month, equivalent to roughly 10-20 hour video.

RUN:
```
conda activate sync
cd AI_Dubbing
python main_app.py
```

todo:

- make a manual correction tab, for change automatic translated texts or base audio if need. (text change done)
- need a good working normaliser metod which is with multilang moduls (change numbers and spacial characters to words, because the finetuned modells not heandle it good)
- good working speaker diarization for segment reworking. (segment reworking done, but it based on speaker diarization, it can use it for podcasts with one speaker)
- non line per line translate, for understand and handle the context (based on chatgpt 4o)
- narration mode (with one narrator only)
- timed SRT upload isnted whisper transcript (manuall edited srt with subtitle editor 4.0.8.)
...

License: MIT



# windows install in docker desktop (With Nvidia GPU support, min. 8-12gb vRAM):

```
git clone https://github.com/sarpba/AI_Dubbing.git
cd AI_Dubbing
cd docker
docker build --build-arg HF_TOKEN=YOUR_HF_TOKEN -t ai_dubbing:0.0.1 .
```

start:
```
docker run --gpus all -p 7860:7860 -p 7861:7861 ai_dubbing:0.0.1
```


