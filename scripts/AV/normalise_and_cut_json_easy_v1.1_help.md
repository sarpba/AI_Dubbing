# normalise_and_cut_json_easy_v1.1_codex – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `AV/normalise_and_cut_json_easy_v1.1.py`

Ez a verzió a generált szinkron szegmenseket az eredeti hang alapján időben igazítja, stabil RMS szinteket állít be és gondoskodik róla, hogy a csúcsok ne legyenek hangosabbak a referenciánál. A fájlok végi, 0,5 másodpercnél hosszabb -50 dB alatti csendet az FFmpeg `silenceremove` szűrője vágja le, miközben a szegmensek elejét továbbra is a VAD-alapú logika rendezi.

## Kötelező beállítás
- `project_name` (pozícionális, alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális beállítások
- `delete_empty` (`--delete_empty`, flag, alapértelmezés: `false`): Ha egy szegmens teljesen csendes marad a VAD szerint, törli a hozzá tartozó audió fájlt.
- `sync_loudness` (`--no-sync-loudness`, flag, alapértelmezés: `true`): Alapból RMS szerint igazítja a szegmenseket és korlátozza a csúcs hangerejüket a referencia szintjéhez. A `--no-sync-loudness` kikapcsolja ezt a lépést.
- `min_db` (`-db`, `--min_db`, option, alapértelmezés: `-40.0`): A hangerő normalizálás alsó korlátja dB-ben. Ezzel akadályozható meg, hogy túl halk referenciát vegyen alapul.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás bekapcsolása.
