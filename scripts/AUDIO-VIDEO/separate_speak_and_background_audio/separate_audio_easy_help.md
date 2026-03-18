# separate_audio_easy

**Futtatási környezet:** `demucs`  
**Belépési pont:** `AUDIO-VIDEO/separate_speak_and_background_audio/separate_audio_easy.py`

## Mit csinál?
Az eredeti filból kivont audio filet beszédre és háttér zenére/sound effectekre bontja.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project` (opció;  kapcsoló: `-p`, `--project`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `device` (opció;  kapcsoló: `--device`; alapértelmezés: `cuda`): A futtatás eszköze, például `cpu`, `cuda`, `cuda:0` vagy `mps`.
- `models` (opció;  kapcsoló: `--models`; alapértelmezés: `htdemucs,mdx_extra`): A használható szeparáló modellek listája, vesszővel elválasztva.
- `chunk_size` (opció;  kapcsoló: `--chunk_size`; alapértelmezés: `5.0`): A szeparálás darabolási mérete.
- `chunk_overlap` (opció;  kapcsoló: `--chunk_overlap`; alapértelmezés: `10.0`): Az egymást követő feldolgozási darabok átfedése.
- `non_speech_silence` (kapcsoló;  kapcsoló: `--non_speech_silence`; alapértelmezés: `false`): A nem beszéd jellegű részeket hangsúlyosabban csendesíti vagy külön kezeli. Alapállapotban ki van kapcsolva.
- `background_blend` (opció;  kapcsoló: `--background_blend`; alapértelmezés: `0.5`): A háttér és a beszéd szétválasztásánál használt keverési arány.
- `keep_full_audio` (kapcsoló;  kapcsoló: `--keep_full_audio`; alapértelmezés: `false`): Megtartja a teljes hosszúságú köztes vagy referenciahangot is. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
