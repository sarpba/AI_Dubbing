Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n demucs
```

Anakonda (nem minikonda) Környezet építése
```bash
conda create -n demucs python=3.10 -y
conda activate demucs
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install demucs numpy
```