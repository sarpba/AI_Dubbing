Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n parakeet
```

```
sudo apt update
sudo apt install build-essential
```

Anakonda (nem minikonda) Környezet építése

```bash
conda create -n parakeet python=3.10 -y
conda activate parakeet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nemo_toolkit[asr]
pip install librosa soundfile
pip install cuda-python>=12.3
conda install -c conda-forge libstdcxx-ng
```
