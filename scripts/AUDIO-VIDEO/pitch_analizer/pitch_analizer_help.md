# pitch_analizer

**Futtatási környezet:** `sync`  
**Belépési pont:** `AUDIO-VIDEO/pitch_analizer/pitch_analizer.py`

## Mit csinál?
Narrátor és translated_splits audiók hangmagasságának összehasonlítása megadott toleranciával.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.
- `narrator` (opció;  kapcsoló: `--narrator`; alapértelmezés: nincs): A narrátor referenciahangjának fájlneve vagy elérési útja, amihez a script a generált szegmenseket hasonlítja.

## Opcionális paraméterek
- `tolerance` (opció;  kapcsoló: `--tolerance`; alapértelmezés: `20.0`): Megengedett eltérés a pitch vagy más összehasonlított mérőszám esetén.
- `min_frequency` (opció;  kapcsoló: `--min-frequency`; alapértelmezés: `60.0`): A vizsgálat alsó frekvenciahatára.
- `max_frequency` (opció;  kapcsoló: `--max-frequency`; alapértelmezés: `400.0`): A vizsgálat felső frekvenciahatára.
- `delete_outside` (kapcsoló;  kapcsoló: `--delete-outside`; alapértelmezés: `false`): Törli azokat a fájlokat, amelyek a megengedett tartományon kívül esnek. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
