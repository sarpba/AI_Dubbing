
# AI Dubbing

## Áttekintés (HU)
Az AI Dubbing egy eszközkészlet videók és hanganyagok többnyelvű szinkronizálásához. A folyamat WhisperX-alapú átiratozást, opcionális beszélőszétválasztást és többféle TTS modellt kombinál a végső kimenet előállításához.

## Overview (EN)
AI Dubbing is a toolkit for multilingual dubbing of videos and audio. The pipeline combines WhisperX-based transcription, optional speaker diarization, and multiple TTS models to produce the final track.

## Telepítési és környezeti útmutató (HU)
1. Telepítsd az alap `sync` conda környezetet a [sync-linux telepítési útmutató](ENVIROMENTS/sync-linux.md) lépései szerint; ez a környezet a projekt fő függőségeit tartalmazza.
2. Ha speciális TTS vagy kísérleti modelleket is futtatnál, állíts be külön környezeteket az alábbi dokumentumok alapján:
   - [f5-tts-linux telepítési útmutató](ENVIROMENTS/f5-tts-linux.md)
   - [parakeet-linux telepítési útmutató](ENVIROMENTS/parakeet-linux.md)
   - [vibevoice-linux telepítési útmutató](ENVIROMENTS/vibevoice-linux.md)
3. A fenti leírások feltételezik, hogy a repó az `AI_Dubbing` könyvtár alatt érhető el, és hogy Anaconda/Miniconda már telepítve van.

## Installation and Environment Setup (EN)
1. Install the base `sync` conda environment by following the [sync-linux setup guide](ENVIROMENTS/sync-linux.md); it contains the core dependencies required to run the project.
2. If you need specialized TTS or experimental models, provision additional environments as described in:
   - [f5-tts-linux setup guide](ENVIROMENTS/f5-tts-linux.md)
   - [parakeet-linux setup guide](ENVIROMENTS/parakeet-linux.md)
   - [vibevoice-linux setup guide](ENVIROMENTS/vibevoice-linux.md)
3. These guides assume the repository is available under `AI_Dubbing` and that Anaconda/Miniconda is already installed on your system.

## Modell- és API-előkészítés (HU)
- Frissítsd a `whisperx/alignment.py` fájlt, és állíts be pontosabb alapértelmezett igazítási modelleket, hogy javuljon az időzítés pontossága.
- Másold a saját `model.pt`, `vocab.txt` és `model_conf.json` fájljaidat a megfelelő `TTS/XXX` alkönyvtárba; sablonfájlokat a `TTS` mappában találsz.
- (Opcionális) Hozz létre Hugging Face fiókot, fogadd el a Pyannote Speaker Diarization 3.1 licencét, majd generálj és tárolj biztonságosan egy olvasási API-kulcsot: https://huggingface.co/pyannote/speaker-diarization-3.1
- Regisztrálj DeepL fiókot, aktiváld az ingyenes API-előfizetést, és készíts API-kulcsot (kb. 500 000 karakter/hó, ~10–20 óra videó).

## Model and API Preparation (EN)
- Update `whisperx/alignment.py` to point to more accurate default alignment models; this improves timeline precision during transcription.
- Copy your `model.pt`, `vocab.txt`, and `model_conf.json` into the appropriate `TTS/XXX` subdirectory. Configuration templates are available in the `TTS` folder.
- (Optional) Create a Hugging Face account, accept the Pyannote Speaker Diarization 3.1 license, and securely store a read-only API token: https://huggingface.co/pyannote/speaker-diarization-3.1
- Register for a DeepL account, enable the free API tier, and generate an API key (500,000 characters/month, roughly 10–20 hours of video).

## Alkalmazás futtatása (HU)
```bash
conda activate sync
cd AI_Dubbing
python main_app.py
```

## Running the Application (EN)
```bash
conda activate sync
cd AI_Dubbing
python main_app.py
```

## Licenc / License
MIT

