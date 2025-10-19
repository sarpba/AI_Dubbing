# unpack_srt_from_mkv_easy – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `AV/unpack_srt_from_mkv_easy.py`

A szkript a feltöltött videófájlba ágyazott feliratokat (`.srt`) bontja ki a projekt számára. Az eredményt a `subtitles` (vagy konfigurációban megadott) almappába menti.

## Kötelező beállítás
- `project_name` (pozícionális, alapértelmezés: nincs): A `workdir` alatti projekt neve, amelynek feltöltött videójából a feliratokat ki kell nyerni.

## Opcionális beállítás
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás bekapcsolása.
