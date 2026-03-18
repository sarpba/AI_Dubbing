# audio-copy-helper

**Futtatási környezet:** `sync`  
**Belépési pont:** `AUDIO-VIDEO/audio_copy_helper/audio-copy-helper.py`

## Mit csinál?
Az upload mappába érkező audió fájlokat 44,1 kHz-es WAV formátumba alakítja, majd a kiválasztott projekt mappákba másolja őket.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `extracted_audio` (kapcsoló;  kapcsoló: `--extracted_audio`; alapértelmezés: `false`): A konvertált hangfájlt az `extracted_audio` mappába másolja. Alapállapotban ki van kapcsolva.
- `separated_audio_background` (kapcsoló;  kapcsoló: `--separated_audio_background`; alapértelmezés: `false`): A konvertált hangfájlt a háttérhang mappájába másolja. Alapállapotban ki van kapcsolva.
- `separated_audio_speech` (kapcsoló;  kapcsoló: `--separated_audio_speech`; alapértelmezés: `false`): A konvertált hangfájlt a beszédhang mappájába másolja. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
