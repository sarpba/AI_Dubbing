# AI_sync
semi-automatic video synchronization

At the moment It run on linux only with anaconda.

Prepare AI_sync anaconda enviroment:
conda  create -n sync python=3.10 && conda activate sync
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda install git
pip install git+https://github.com/m-bain/whisperx.git
git clone 

Prepare F5-TTS anaconda enviroment:
```
conda  create -n F5-TTS python=3.10 && conda activate F5-TTS
conda install git
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/SWivid/F5-TTS.git
```
