# SAM_audio – beszéd/háttér szétválasztás SAM-Audio backenddel
**Runtime:** `sam-audio`  
**Entry point:** `AUDIO-VIDEO/SAM_audio/SAM_audio.py`

## Overview
Ez a modul a projekt `1.5_extracted_audio` mappájából származó audiókat választja szét
beszéd és háttér sávokra a SAM-Audio ONNX interface (onnx_inference.py) használatával.

## Required Parameters
- `--project-name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatti projekt neve.

## Optional Parameters
- `--prompt` (option, alapértelmezés: `speech`): Text prompt, amely megadja a kiválasztandó hangot.
- `--device` (option, alapértelmezés: `cuda`): Futási eszköz (`cuda` vagy `cpu`).
- `--onnx-script` (option, alapértelmezés: `onnx_inference.py`): ONNX interface fájl elérési útja.
- `--onnx-model-dir` (option, alapértelmezés: `onnx_models`): ONNX modelleket tartalmazó mappa.
- `--onnx-steps` (option, alapértelmezés: `16`): ODE lépések száma az ONNX pipeline-ban.
- `--sample-rate` (option, alapértelmezés: `48000`): FFmpeg konverziós mintavétel (Hz).
- `--mono` (flag, alapértelmezés: `false`): Mono WAV konverzió erőltetése.
- `--force-convert` (flag, alapértelmezés: `false`): Minden bemenet WAV-ra konvertálása.
- `--keep-temp` (flag, alapértelmezés: `false`): Ideiglenes WAV fájlok megtartása.
- `--predict-spans` (flag, alapértelmezés: `false`): Automatikus span predikció engedélyezése.
- `--reranking-candidates` (option, alapértelmezés: `1`): Reranking jelöltek száma.
- `--span-threshold` (option, alapértelmezés: `0.3`): Span predikció küszöbértéke.
- `--chunk-seconds` (option, alapértelmezés: `60`): Feldolgozás darabolása másodpercben (0 = teljes fájl).
- `--debug` (flag, alapértelmezés: `false`): Részletes naplózás engedélyezése.

## Outputs
- `2_separated_audio_speech/<fajlnev>_speech.wav`
- `2_separated_audio_background/<fajlnev>_non_speech.wav`

## Error Handling / Tips
- Az ffmpeg-nek elérhetőnek kell lennie a PATH-on.
- Az ONNX interface betöltéséhez add meg a `--onnx-script` fájlt (a `matbee/sam-audio-small-onnx` repóból).
- A `--predict-spans` és a `--reranking-candidates` csak akkor működik, ha a megfelelő ONNX modellek is jelen vannak a `--onnx-model-dir` mappában (PEAFrame/CLAP).
