# soniox – aszinkron STT pipeline
**Futtatási környezet:** `sync`  
**Belépési pont:** `soniox.py`

## Áttekintés
 Az eszköz a Soniox hivatalos REST alapú aszinkron STT API-ját hívja meg a kiválasztott projekt `separated_audio_speech` mappájában található hangfájlokra. Minden fájlhoz külön job készül: a kliens feltölti a hangot a Soniox felhőbe, létrehozza a transzkripciót, ciklikusan lekérdezi a státuszt, majd a kész eredményt normalizált `word_segments` listaként **és automatikusan generált mondatszegmensek** (`segments`) formájában menti az eredeti fájl mellé (azonos fájlnévvel, `.json` kiterjesztésben). A JSON tartalmazza a legfontosabb metaadatokat (Soniox file/transcription ID, státusz, létrehozási idő). A Soniox API eredeti (nyers) JSON válasza párhuzamosan `<fájlnév>.soniox_raw.txt` állományba is elmentésre kerül.

## Kötelező paraméterek
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatt található projektmappa, amelynek `separated_audio_speech` almappáját dolgozza fel.

## Opcionális paraméterek
- `model` (`--model`, option, alapértelmezés: `stt-async-v3`): Soniox modell azonosító. Igény szerint felülírható bármely támogatott modellre.
- `min_speakers` (`--min-speakers`, option, alapértelmezés: `0`): Kompatibilitási flag. A Soniox REST API jelenleg nem kezeli ezt, figyelmeztetést írunk ki.
- `max_speakers` (`--max-speakers`, option, alapértelmezés: `0`): Kompatibilitási flag. A Soniox REST API jelenleg nem kezeli ezt, figyelmeztetést írunk ki.
- `candidate_speaker` (`--candidate-speaker`, option, ismételhető, alapértelmezés: nincs): Kompatibilitási flag; a Soniox REST API nem használja.
- `no_diarize` (`--no-diarize`, flag, alapértelmezés: `false`): Kikapcsolja a globális diarizációt (alapból be van kapcsolva).
- `speaker_identification` (`--speaker-identification`, flag, alapértelmezés: `false`): Kompatibilitási flag; REST módban jelenleg nincs hatása.
- `api_key` (`--api-key`, option, alapértelmezés: nincs): Soniox API kulcs. Megadáskor a kulcs base64 formában elmentésre kerül a `keyholder.json` fájlba. Alternatívaként használható a `SONIOX_API_KEY` környezeti változó.
- `api_host` (`--api-host`, option, alapértelmezés: nincs): Egyedi Soniox host (pl. privát endpoint) megadása.
- `reference_prefix` (`--reference-prefix`, option, alapértelmezés: `soniox`): Az aszinkron job referencia neve ez alapján épül fel (`<prefix>/<fáljnév>`).
- `poll_interval` (`--poll-interval`, option, alapértelmezés: `5.0`): A státusz lekérdezések közötti várakozási idő másodpercben.
- `timeout` (`--timeout`, option, alapértelmezés: `1800`): Maximális várakozási idő másodpercben. `0` esetén nincs időkorlát.
- `chunk_size` (`--chunk-size`, option, alapértelmezés: `131072`): Kompatibilitási opció (REST módban nincs szerepe).
- `overwrite` (`--overwrite`, flag, alapértelmezés: `false`): Ha aktív, a már létező kimeneti JSON fájlok felülírásra kerülnek, egyébként a szkript átugorja őket.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes logolás bekapcsolása.

## Kimenetek
- Minden bemeneti hangfájl mellé létrejön egy `<fájlnév>.json`, amely tartalmazza:
  - `word_segments`: szó szintű időbélyegek, opcionális pontossági értékkel, beszélő és csatorna mezőkkel.
  - `segments`: automatikusan képzett mondatszegmensek (start, end, text, words, opcionális speaker mezővel), így külön `resegment` lépés nélkül is használhatók a további folyamatok.
  - `speaker_labels`: a Soniox által visszaadott beszélő azonosítók és (ha elérhető) nevek.
  - `metadata`: `file_id`, `reference_name`, `status`, `created` mezők, így könnyen visszakereshető a Soniox portálon.
- Egy `<fájlnév>.soniox_raw.txt` fájl, amely a Soniox API teljes JSON válaszát tartalmazza ember-olvasható formában (debug/nyers ellenőrzéshez).

## Hibakezelés / tippek
- A szkript a Soniox REST API-ját hívja meg `requests` segítségével, ezért nincs szükség a hivatalos Python SDK telepítésére.
- API kulcs hiányában a futás leáll. A kulcs megadható flaggel vagy a `SONIOX_API_KEY` környezeti változóval; mindkét esetben ajánlott a `keyholder.json` titkosított tároló használata.
- Az `--overwrite` hiánya garantálja, hogy ismételt futtatáskor nem sérülnek a korábbi kimenetek (idempotens működés). Ha friss kimenetre van szükség, kapcsold be.
- Hosszabb fájlok feldolgozásához szükség esetén növeld a `--timeout` értékét, illetve csökkentsd/növeld a `--poll-interval`-t a kívánt terhelés függvényében.
- Ha a Soniox fiókod webhookokat vagy fordítási módokat igényel, bővítsd a scriptet a dokumentációban található extra mezőkkel (payload bővítése a `build_transcription_payload` függvényben).
