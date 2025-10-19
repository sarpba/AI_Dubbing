# normalise_and_cut_json_easy_codex – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `AV/normalise_and_cut_json_easy.py`

A szkript a TTS által generált szinkron szegmenseket hangerőben az eredeti beszédhez igazítja, és – ha szükséges – a diarizáció alapján levágja a szegmensek elejét a pontosabb illeszkedés érdekében.

## Kötelező beállítás
- `project_name` (pozícionális, alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális beállítások
- `delete_empty` (`--delete_empty`, flag, alapértelmezés: `false`): Ha egy szegmens teljesen csendes marad a VAD szerint, törli a hozzá tartozó audió fájlt.
- `sync_loudness` (`--no-sync-loudness`, flag, alapértelmezés: `true`): Alapból megpróbálja a generált darabok RMS-ét az eredeti beszédhez igazítani. A `--no-sync-loudness` kikapcsolja ezt a lépést.
- `min_db` (`-db`, `--min_db`, option, alapértelmezés: `-40.0`): A hangerő normalizálás alsó korlátja dB-ben. Ezzel akadályozható meg, hogy túl halk referenciát vegyen alapul.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás bekapcsolása.
