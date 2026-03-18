# soniox

**Futtatási környezet:** `sync`  
**Belépési pont:** `ASR/soniox/soniox.py`

## Mit csinál?
Soniox ASR aszinkron API a projekthez tartozó hangfájlokhoz: word_segments + mondatszegmensek (segments) JSON mentése és a nyers API válaszok .soniox_raw.txt dumpja.

A script a projekt hanganyagából készít átírást vagy újraszegmentált JSON-t, hogy a további fordítási és TTS lépések már strukturált bemenettel dolgozzanak.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `model` (opció;  kapcsoló: `--model`; alapértelmezés: `stt-async-v3`): A használt modell neve vagy azonosítója.
- `diarize` (kapcsoló;  kapcsoló: `--diarize`; alapértelmezés: `false`): Bekapcsolja a beszélőfelismerést, így a kimenet beszélő szerinti információt is tartalmaz. Alapállapotban ki van kapcsolva.
- `api_key` (opció;  kapcsoló: `--api-key`; alapértelmezés: nincs): API kulcs az adott külső szolgáltatáshoz.
- `api_host` (opció;  kapcsoló: `--api-host`; alapértelmezés: nincs): Egyedi hosztnév vagy API cím. Haladó beállítás speciális infrastruktúrához.
- `reference_prefix` (opció;  kapcsoló: `--reference-prefix`; alapértelmezés: `soniox`): Előtag a külső szolgáltatásnál létrehozott jobok vagy referenciák azonosításához.
- `poll_interval` (opció;  kapcsoló: `--poll-interval`; alapértelmezés: `5.0`): A státuszlekérdezések közti várakozás másodpercben.
- `timeout` (opció;  kapcsoló: `--timeout`; alapértelmezés: `1800`): A maximális várakozási idő másodpercben.
- `overwrite` (kapcsoló;  kapcsoló: `--overwrite`; alapértelmezés: `false`): A már meglévő kimeneti fájlokat is újragenerálja vagy felülírja. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
