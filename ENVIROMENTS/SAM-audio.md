Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n sam-audio
```

Anakonda (nem minikonda) Környezet építése

```bash
conda create -n sam-audio python=3.11 -y
conda activate sam-audio
conda install git -y
sudo apt update
sudo apt install ffmpeg -y
```

SAM-Audio ONNX repo (matbee)

```bash
git clone https://huggingface.co/matbee/sam-audio-small-onnx
cd sam-audio-small-onnx
pip install onnxruntime sentencepiece soundfile torchaudio librosa
```

Megjegyzések
- CUDA-s futtatáshoz NVIDIA driver és onnxruntime-gpu szükséges (ha CPU-n futtatod, maradhat az alap onnxruntime).
- Az ONNX modelleket a repo `onnx_models` mappája tartalmazza; ezt add meg a `--onnx-model-dir` paraméterrel.
