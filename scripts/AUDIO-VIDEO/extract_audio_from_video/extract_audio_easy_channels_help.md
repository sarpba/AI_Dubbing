# extract_audio_easy_channels

**Futtatási környezet:** `sync`  
**Belépési pont:** `AUDIO-VIDEO/extract_audio_from_video/extract_audio_easy_channels.py`

## Mit csinál?
Az audio sávokat kinyeri a videó fájlokból, külön csatornákra bontva, vagy egy stereo csatornára muxolva.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project_dir_name` (pozicionális;  kapcsoló: pozicionális; alapértelmezés: nincs): A feldolgozandó projekt könyvtárneve a `workdir` alatt.

## Opcionális paraméterek
- `keep_channels` (kapcsoló;  kapcsoló: `--keep_channels`; alapértelmezés: `false`): Megőrzi a külön csatornákat külön fájlokban, ahelyett hogy egyetlen sztereó kimenet készülne. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
