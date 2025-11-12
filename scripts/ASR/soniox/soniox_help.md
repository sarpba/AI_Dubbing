# soniox – aszinkron STT pipeline
**Futtatási környezet:** `sync`  
**Belépési pont:** `soniox.py`

## Áttekintés
 Az eszköz a Soniox hivatalos REST alapú aszinkron STT API-ját hívja meg a kiválasztott projekt `separated_audio_speech` mappájában található hangfájlokra. Minden fájlhoz külön job készül: a kliens feltölti a hangot a Soniox felhőbe, létrehozza a transzkripciót, ciklikusan lekérdezi a státuszt, majd a kész eredményt normalizált `word_segments` listaként **és automatikusan generált mondatszegmensek** (`segments`) formájában menti az eredeti fájl mellé (azonos fájlnévvel, `.json` kiterjesztésben). A JSON tartalmazza a legfontosabb metaadatokat (Soniox file/transcription ID, státusz, létrehozási idő). A Soniox API eredeti (nyers) JSON válasza párhuzamosan `<fájlnév>.soniox_raw.txt` állományba is elmentésre kerül.

## Kötelező paraméterek
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatt található projektmappa, amelynek `separated_audio_speech` almappáját dolgozza fel.

## Opcionális paraméterek
- `model` (`--model`, option, alapértelmezés: `stt-async-v3`): Soniox modell azonosítója.
- `diarize` (`--diarize`, flag, alapértelmezés: `false`): Speaker diarizáció bekapcsolása.
- `api_key` (`--api-key`, option, alapértelmezés: nincs): Soniox API kulcs megadása/mentése (`SONIOX_API_KEY` környezeti változó is használható).
- `api_host` (`--api-host`, option, alapértelmezés: nincs): Egyedi Soniox host (pl. privát endpoint).
- `reference_prefix` (`--reference-prefix`, option, alapértelmezés: `soniox`): A Soniox async job `client_reference_id` előtagja.
- `poll_interval` (`--poll-interval`, option, alapértelmezés: `5.0`): Státuszlekérdezések közti várakozás másodpercben.
- `timeout` (`--timeout`, option, alapértelmezés: `1800`): Max. várakozási idő (0 = végtelen).
- `overwrite` (`--overwrite`, flag, alapértelmezés: `false`): Meglévő kimeneti JSON felülírása.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes logolás.

## Kimenetek
- Minden bemeneti hangfájl mellé létrejön egy `<fájlnév>.json`, amely tartalmazza:
  - `word_segments`: szó szintű időbélyegek, opcionális pontossági értékkel, beszélő és csatorna mezőkkel.
  - `segments`: automatikusan képzett mondatszegmensek (start, end, text, words, opcionális speaker mezővel), így külön `resegment` lépés nélkül is használhatók a további folyamatok.
  - `speaker_labels`: a Soniox által visszaadott beszélő azonosítók és (ha elérhető) nevek.
  - `metadata`: `file_id`, `reference_name`, `status`, `created` mezők, így könnyen visszakereshető a Soniox portálon.
- Egy `<fájlnév>.soniox_raw.txt` fájl, amely a Soniox API teljes JSON válaszát tartalmazza ember-olvasható formában (debug/nyers ellenőrzéshez).

## Hibakezelés / tippek
- A Soniox REST API maximum 500 MB-os bemenetet fogad. Ha egy fájl ennél nagyobb, a szkript automatikusan MP3-ra tömöríti (ffmpeg szükséges), majd a tömörített példányt tölti fel.
- API kulcs hiányában a futás leáll; add meg flaggel vagy `SONIOX_API_KEY` változóval, így elmentődik a `keyholder.json`-ba.
- Az `--overwrite` hiánya garantálja, hogy ismételt futáskor nem írjuk felül a meglévő JSON-okat.
- Hosszabb fájlok esetén növeld a `--timeout`-ot, vagy állítsd a `--poll-interval`-t a kívánt terheléshez.
