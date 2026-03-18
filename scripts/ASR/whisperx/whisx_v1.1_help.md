# whisx_v1.1

**Futtatási környezet:** `whisperx`  
**Belépési pont:** `ASR/whisperx/whisx_v1.1.py`

## Mit csinál?
WhisperX modell használata a beszéd szöveggé alakítására, több GPU támogatással, diarizációval és egyéni, Parakeet-stílusú szegmentálással.

A script a projekt hanganyagából készít átírást vagy újraszegmentált JSON-t, hogy a további fordítási és TTS lépések már strukturált bemenettel dolgozzanak.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `gpus` (opció;  kapcsoló: `--gpus`; alapértelmezés: nincs): A használható GPU-k listája vagy száma. Több GPU esetén a script párhuzamos feldolgozást is használhat.
- `hf_token` (opció;  kapcsoló: `--hf_token`; alapértelmezés: nincs): Hugging Face token a védett modellek vagy diarizációs komponensek eléréséhez.
- `language` (opció;  kapcsoló: `--language`; alapértelmezés: nincs): A feldolgozás nyelve vagy a kész kimenet nyelvi kódja.
- `max_pause` (opció;  kapcsoló: `--max-pause`; alapértelmezés: `0.6`): Legfeljebb ekkora szünetet hagy a script egy szegmensen belül. Nagyobb érték hosszabb mondatrészeket eredményezhet.
- `timestamp_padding` (opció;  kapcsoló: `--timestamp-padding`; alapértelmezés: `0.2`): Ennyi időt ad hozzá a szegmensek elejéhez és végéhez a kényelmesebb vágás érdekében.
- `max_segment_duration` (opció;  kapcsoló: `--max-segment-duration`; alapértelmezés: `11.5`): A szegmensek maximális hossza másodpercben.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
