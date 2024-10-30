# a semi-automatic video synchronization system 
WORKING IN PROGRESS... not sure it work flavesly when cloned. Loock back a few hour or day leater.


The project uses the following AI models:

OpenAI Whisper, 
Pyannote Speaker Diarization, 
Facebookresearch demucs MDXnet, 
SWivid F5-TTS

It's a hobby project. I'll just a little maintan.

At the moment It run on linux only with anaconda.

Prepare F5-TTS anaconda enviroment:
```
conda create -n F5-TTS python=3.10 && conda activate F5-TTS
conda install git
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/SWivid/F5-TTS.git
```

Prepare AI_sync anaconda enviroment:
```
conda  create -n sync python=3.10 && conda activate sync
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda install git
pip install git+https://github.com/m-bain/whisperx.git
git clone https://github.com/sarpba/AI_sync.git
cd AI_sync
pip install -r requirements.txt
```
Copy your F5-TTS model.pt and vocab.txt into the TTS directory. 

Create a Hugging Face account, and accept the licenses for the following:

Pyannote Speaker Diarization 3.1 Generate a read API key and save it securely. https://huggingface.co/pyannote/speaker-diarization-3.1
Register for a DeepL account, activate the API with a free subscription, and generate an API key as well. This free tier includes up to 500,000 characters per month, equivalent to roughly 6-10 full movies.

RUN:
```
cd AI_sync
python main_app.py
```

todo:

- make a manual correction tab, for change automatic translate or base audio fi need.
- need a good working normaliser metod which is with multilang moduls (change numbers and spacial characters to words, because the finetuned modells not heandle it good)
- change the TTS run method a faster on (with multigpu support)
...

License: MIT