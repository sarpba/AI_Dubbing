# evenlabs – konfigurációs útmutató

**Futtatási környezet:** `sync`
**Belépési pont:** `elevenlabs.py`

A szkript az Evenlabs speech-to-text API-t hívja meg, és a projekt `separated_audio_speech` mappájában található hangfájlokat szó szintű időbélyegekkel ellátott JSON-átirattá alakítja. A válaszban kapott szólistát saját logikával szegmentálja, ha az API nem szolgáltat mondatszegmenseket.

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatti projektmappa neve, amelynek hanganyagát feldolgozza.

## Opcionális beállítások
- `language` (`--language`, option, alapértelmezés: nincs): ISO nyelvkód (pl. `en`, `hu`). Ha nincs megadva, az Evenlabs automatikus nyelvfelismerésével dolgozik.
- `model` (`--model`, option, alapértelmezés: `eleven_multilingual_v2`): Az Evenlabs ASR modell azonosítója.
- `api_url` (`--api-url`, option, alapértelmezés: `https://api.elevenlabs.io/v1/speech-to-text`): A használni kívánt Evenlabs STT végpont.
- `api_key` (`--api-key`, option, alapértelmezés: nincs): Evenlabs API kulcs. Megadáskor elmenti base64 formában a `keyholder.json` fájlba. Alternatíva az `EVENLABS_API_KEY` környezeti változó.
- `max_pause` (`--max-pause`, option, alapértelmezés: `0.8`): A saját szegmentálás során megengedett maximális szünet két szó között (mp). Nagyobb érték kevésbé bontja fel a hosszabb csöndeket.
- `timestamp_padding` (`--timestamp-padding`, option, alapértelmezés: `0.1`): Extra margó, amelyet a szavak időbélyegeihez ad hozzá (mp), hogy ne vágja le a szókezdetet/szótvégét.
- `max_segment_duration` (`--max-segment-duration`, option, alapértelmezés: `14.0`): Egy szegmens legnagyobb hossza másodpercben. `0` esetén nincs korlát.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás, figyelmeztetések megjelenítése.

> **Megjegyzés:** A szkript minden feldolgozott hangfájllal azonos nevű `.json` fájlt hoz létre ugyanabban a mappában.
