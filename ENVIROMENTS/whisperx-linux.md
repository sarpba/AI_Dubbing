Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n whisperx
```

Anakonda (nem minikonda) Környezet építése
```bash
conda create -n whisperx python=3.10 -y
conda activate whisperx
conda install -y -c conda-forge ffmpeg sox libsndfile rust git pkg-config
pip install --upgrade pip
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install "triton>=3.3.0"
pip install "nvidia-cudnn-cu12>=9.1.0"
```

Környezeti változók beállítása
```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python - << 'PY'
import site, glob, os
sp = site.getsitepackages()[0]
print(glob.glob(os.path.join(sp, "nvidia", "cudnn", "lib"))[0])
PY
)"
```

Whisperx telepítése
```bash
pip install -U whisperx
```