# normalise_and_cut_json_easy_v1.1

**Futtatási környezet:** `sync`  
**Belépési pont:** `AUDIO-VIDEO/normalize_and_cut_tts_files/normalise_and_cut_json_easy_v1.1.py`

## Mit csinál?
A TTS-el generált audiókat az eredeti hangsáv alapján időben és hangerőben igazítja, a végeken automatikusan eltávolítja a hosszabb csendet FFmpeg silenceremove-val.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project_name` (pozicionális;  kapcsoló: pozicionális; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `delete_empty` (kapcsoló;  kapcsoló: `--delete_empty`; alapértelmezés: `false`): Eltávolítja azokat a kimeneti hangfájlokat, amelyek üresnek vagy használhatatlannak bizonyulnak. Alapállapotban ki van kapcsolva.
- `sync_loudness` (kapcsoló;  kapcsoló: `--no-sync-loudness`; alapértelmezés: `true`): Alapból bekapcsolt hangerő-illesztés. A script a generált hangot a referencia hangerejéhez igazítja. Alapállapotban be van kapcsolva; a negatív kapcsolóval kikapcsolható.
- `min_db` (opció;  kapcsoló: `-db`, `--min_db`; alapértelmezés: `-40.0`): Alsó hangerőkorlát decibelben. Segít elkerülni, hogy túl halk részekhez igazodjon a normalizálás.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
