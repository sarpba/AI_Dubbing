Esetleges korábbi telepítés törlése

```bash
conda deactivate
conda env remove -n higgs-tts-gradio
```

Anakonda (nem minikonda) környezet építése

Ajánlott, reprodukálható út a lockolt környezettel:

```bash
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
conda env create -f environment.lock.yml
conda activate higgs-tts-gradio
```

Fejlesztői út a rövidebb, lazább `environment.yml` fájllal:

```bash
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
conda env create -f environment.yml
conda activate higgs-tts-gradio
```

Meglévő környezet frissítése:

```bash
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
conda env update -n higgs-tts-gradio -f environment.yml --prune
```

A környezet fő csomagjai

- `python=3.12`
- `ffmpeg`
- `gradio==6.19.0`
- `python-dotenv==1.2.2`
- `pandas==3.0.3`
- `numpy==2.5.1`
- `soundfile==0.14.0`
- `huggingface-hub==1.22.0`
- `torch==2.12.1`
- `torchaudio==2.11.0`
- `transformers==5.13.0`
- `accelerate==1.14.0`
- `safetensors==0.8.0`

Miért nem közvetlenül a `bosonai/higgs-tts-3-4b` modellt tölti

A `bosonai/higgs-tts-3-4b` repo nem tartalmazza a közvetlen Transformers betöltéshez szükséges `auto_map` custom kódot, ezért a lokális futtatás a Transformers-kompatibilis:

```text
multimodalart/higgs-audio-v3-tts-4b-transformers
```

modellre támaszkodik.

Gyors ellenőrzés telepítés után

```bash
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
python -m py_compile app.py
python -c "import app; print(app.check_local_backend('bosonai/higgs-tts-3-4b'))"
```

Ez ellenőrzi, hogy:

- az importok felállnak;
- a Gradio app betölthető;
- a local backend képes modellt választani;
- a CUDA/CPU mód detektálása működik.

Indítás

```bash
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
conda activate higgs-tts-gradio
python app.py
```

Vagy:

```bash
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
./launch_gradio.sh
```

Alapértelmezett cím:

```text
http://127.0.0.1:7860
```

Opcionális `.env` fájl

```bash
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
cp .env.example .env
```

Fontos változók:

- `HIGGS_MODEL`: UI-ban megjelenő modell alias, alapból `bosonai/higgs-tts-3-4b`
- `HIGGS_LOCAL_MODEL`: a ténylegesen használt local Transformers modell, alapból `multimodalart/higgs-audio-v3-tts-4b-transformers`
- `GRADIO_SERVER_NAME`: bind cím
- `GRADIO_SERVER_PORT`: port

Referenciahang kezelés

A környezethez tartozó `app.py` és az AI_Dubbing projektben készített `scripts/TTS/higgs/higgs-tts.py` ugyanazt a rövid referenciahang-kezelési elvet használja:

- ha a referencia legalább `2.0` másodperc, változatlan marad;
- ha rövidebb mint `2.0` másodperc, ismétléssel legalább `4.0` másodperc környékére bővül;
- a stereo referencia automatikusan mono-ra keveredik;
- a cél a stabilabb voice cloning rövid szegmenseknél.

Újratelepítési teszt

```bash
conda env remove -n higgs-tts-gradio-reinstall-test -y || true
cd /home/sarpba/Sajat_programok/higgs-tts-3-4b
conda env create -n higgs-tts-gradio-reinstall-test -f environment.lock.yml
conda run -n higgs-tts-gradio-reinstall-test python -m py_compile app.py
conda run -n higgs-tts-gradio-reinstall-test python -c "import torch, torchaudio, transformers, gradio; print(torch.__version__, torchaudio.__version__, transformers.__version__, gradio.__version__)"
conda run -n higgs-tts-gradio-reinstall-test python -c "import app; print(app.preview_prompt('Teszt.', 'anger', '', 'none', 'none', 'none', 'none', '', ''))"
conda env remove -n higgs-tts-gradio-reinstall-test -y
```

Megjegyzés

Az AI_Dubbing projekt új `scripts/TTS/higgs/higgs-tts.py` modulja ezt a `higgs-tts-gradio` környezetet várja, ezért a JSON descriptorban is ez az environment név szerepel.
