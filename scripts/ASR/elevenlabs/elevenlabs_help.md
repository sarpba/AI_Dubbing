# elevenlabs

**Futtatási környezet:** `sync`  
**Belépési pont:** `ASR/elevenlabs/elevenlabs.py`

## Mit csinál?
Evenlabs ASR API használata a projekthez tartozó hangfájlok feldolgozására, normalizált szó szintű (word_segments) JSON kimenettel.

A script a projekt hanganyagából készít átírást vagy újraszegmentált JSON-t, hogy a további fordítási és TTS lépések már strukturált bemenettel dolgozzanak.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `language` (opció;  kapcsoló: `--language`; alapértelmezés: nincs): A feldolgozás nyelve vagy a kész kimenet nyelvi kódja.
- `model` (opció;  kapcsoló: `--model`; alapértelmezés: `scribe_v1_experimental`): A használt modell neve vagy azonosítója.
- `api_url` (opció;  kapcsoló: `--api-url`; alapértelmezés: `https://api.elevenlabs.io/v1/speech-to-text`): Az API végpontja. Általában csak akkor módosítsd, ha saját vagy eltérő szolgáltatói endpointot használsz.
- `api_key` (opció;  kapcsoló: `--api-key`; alapértelmezés: nincs): API kulcs az adott külső szolgáltatáshoz.
- `no_diarize` (kapcsoló;  kapcsoló: `--no-diarize`; alapértelmezés: `false`): Kikapcsolja a beszélőfelismerést. Alapállapotban a script diarizációval dolgozik. Alapállapotban ki van kapcsolva; a kapcsoló aktiválásakor a script letilt egy belső funkciót.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
