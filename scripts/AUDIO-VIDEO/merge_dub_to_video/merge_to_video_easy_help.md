# merge_to_video_easy

**Futtatási környezet:** `sync`  
**Belépési pont:** `AUDIO-VIDEO/merge_dub_to_video/merge_to_video_easy.py`

## Mit csinál?
A korábban létrehozott szinkron audiót összemuxolja a forrás videóval. Valamint ha volt felirat, feltöltve, vagy az eredeti videóból kinyerve, akkor azt is beágyazza a videóba.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project_name` (pozicionális;  kapcsoló: pozicionális; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.
- `language` (opció;  kapcsoló: `-lang`, `--language`; alapértelmezés: nincs): A kész videó hangsávjához és feliratához tartozó nyelvi kód.

## Opcionális paraméterek
- `only_new_audio` (kapcsoló;  kapcsoló: `--only-new-audio`; alapértelmezés: `false`): Csak az új hangsávot muxolja a videóba, és a többi kiegészítő lépést minimalizálja. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
