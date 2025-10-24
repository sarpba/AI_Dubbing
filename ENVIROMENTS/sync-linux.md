Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n sync
```

Anakonda (nem minikonda) Környezet építése

```bash
conda  create -n sync python=3.10 -y && conda activate sync
conda install git
git clone https://github.com/sarpba/AI_Dubbing.git
cd AI_Dubbing_new
pip install -r requirements.txt
sudo apt update
sudo apt install ffmpeg
sudo apt install gcc
sudo apt install mkvtoolnix mkvtoolnix-gui
```

trash:
#pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
#pip install git+https://github.com/m-bain/whisperx.git
#conda install cudatoolkit=11.8 cudnn=8.*
#sudo wget -qO - https://mkvtoolnix.download/gpg-pub-moritzbunkus.gpg | gpg --dearmor -o /etc/apt/trusted.gpg.d/mkvtoolnix.gpg
#echo "deb [arch=amd64] https://mkvtoolnix.download/ubuntu/ $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/mkvtoolnix.list