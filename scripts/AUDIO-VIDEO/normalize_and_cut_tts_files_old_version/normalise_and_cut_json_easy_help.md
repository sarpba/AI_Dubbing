# normalise_and_cut_json_easy

**Futtatási környezet:** `sync`  
**Belépési pont:** `AUDIO-VIDEO/normalize_and_cut_tts_files_old_version/normalise_and_cut_json_easy.py`

## Mit csinál?
A TTS-el generált audiok hangerejét az eredeti audio hangerejéhez igazítja, valamint a beszédszegmensek alapján levágja a generált szinkron darabok elejét a pontosabb szinkron érdekében.

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
