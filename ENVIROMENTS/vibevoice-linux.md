Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n vibevoice
```

Anakonda (nem minikonda) Környezet építése

```bash
git clone https://github.com/sarpba/VibeVoice.git
cd VibeVoice
conda create --name vibevoice python=3.9 -y
conda activate vibevoice
pip install -e .
pip install openai-whisper
pip install transformers accelerate
pip install Levenshtein num2words
```