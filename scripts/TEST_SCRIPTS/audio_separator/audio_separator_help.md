# audio_separator

**Futtatási környezet:** `demucs`  
**Belépési pont:** `AUDIO-VIDEO/audio_separator/audio_separator.py`

## Mit csinál?
Ez a modul a `python-audio-separator` csomagra építve választja szét a projekt `extracted_audio` fájljait beszéd- és háttérsávra.

Két külön workflow-t indít:
- speaker workflow: alapból `htdemucs_ft.yaml` + `UVR-MDX-NET-Voc_FT.onnx`
- background workflow: alapból `htdemucs_ft.yaml`

A speaker oldali modellkombináció szabadon cserélhető, így például `MDX23C-8KFFT-InstVoc_HQ.ckpt` is megadható az MDX-Net helyett.

A repo korábbi, filmes anyagokon bevált Demucs/MDX irányához igazodva a háttérsáv utófeldolgozást is alkalmaz: a modell által becsült háttér és az eredeti mixből kivont residual háttér alacsony arányú blendjét használja. Ez kevésbé tompa eredményt ad, mint a nyers stem export.

## Kötelező paraméterek
- `project` (opció; kapcsoló: `-p`, `--project`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Fontos opcionális paraméterek
- `speech_models` (opció; kapcsoló: `--speech_models`; alapértelmezés: `htdemucs_ft.yaml,UVR-MDX-NET-Voc_FT.onnx`): A beszédsávhoz használt modellek listája.
- `background_models` (opció; kapcsoló: `--background_models`; alapértelmezés: `htdemucs_ft.yaml`): A háttérsávhoz használt modellek listája.
- `speech_ensemble_algorithm` (opció; kapcsoló: `--speech_ensemble_algorithm`; alapértelmezés: `avg_wave`): A speaker workflow ensemble algoritmusa.
- `background_ensemble_algorithm` (opció; kapcsoló: `--background_ensemble_algorithm`; alapértelmezés: `avg_wave`): A background workflow ensemble algoritmusa.
- `background_blend` (opció; kapcsoló: `--background_blend`; alapértelmezés: `0.2`): A modell háttérsávja és a residual háttér keverési aránya. Alacsony érték természetesebb filmes hátteret ad.
- `non_speech_silence` (kapcsoló; kapcsoló: `--non_speech_silence`; alapértelmezés: `false`): Tiszta beszédrészeknél vagy mindig lenémítja a background sávot.
- `speech_acceleration` (opció; kapcsoló: `--speech_acceleration`; alapértelmezés: `auto`): A beszéd workflow gyorsítási módja.
- `background_acceleration` (opció; kapcsoló: `--background_acceleration`; alapértelmezés: `auto`): A háttér workflow gyorsítási módja. `auto` esetén induláskor megnézi a GPU-k számát, és ha van második GPU, azon fut; ha csak egy GPU érhető el, CPU-ra vált fallbackként.
- `run_sequentially` (kapcsoló; kapcsoló: `--run_sequentially`; alapértelmezés: `false`): Ha be van kapcsolva, a két workflow nem párhuzamosan fut.
- `keep_intermediate_stems` (kapcsoló; kapcsoló: `--keep_intermediate_stems`; alapértelmezés: `false`): Megtartja a background workflow köztes stem-fájljait.
- `model_file_dir` (opció; kapcsoló: `--model_file_dir`; alapértelmezés: nincs): Opcionális modell-cache könyvtár.
- `chunk_duration` (opció; kapcsoló: `--chunk_duration`; alapértelmezés: nincs): Hosszú fájlok feldarabolása másodpercben.
- `debug` (kapcsoló; kapcsoló: `--debug`; alapértelmezés: `false`): Részletes hibakereső naplózás.

## Példák
```bash
python scripts/AUDIO-VIDEO/audio_separator/audio_separator.py -p demo
```

```bash
python scripts/AUDIO-VIDEO/audio_separator/audio_separator.py \
  -p demo \
  --speech_models MDX23C-8KFFT-InstVoc_HQ.ckpt,3_HP-Vocal-UVR.pth \
  --background_models htdemucs.yaml
```

## Megjegyzés
A futtató környezetben telepítve kell lennie a `audio-separator` csomagnak. Ha nincs jelen, a script egyértelmű hibaüzenettel leáll.
