# extract_audio_easy_channels – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `AV/extract_audio_easy_channels.py`

A szkript a `workdir/<projekt>/upload` mappában található első videófájlból kinyeri az audiósávot. Igény szerint minden csatornát külön WAV fájlként ment, és elmenti az FFprobe által szolgáltatott stream metaadatokat is.

## Kötelező beállítás
- `project_dir_name` (pozícionális, alapértelmezés: nincs): A projekt neve a `workdir` alatt. A szkript ebből határozza meg az `upload` és `extracted_audio` almappákat.

## Opcionális beállítások
- `keep_channels` (`--keep_channels`, flag, alapértelmezés: `false`): Többcsatornás forrásnál minden csatornát külön fájlba bont. Ha nincs bekapcsolva, vagy csak sztereó a bemenet, egyetlen sztereó WAV készül.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózást kér a `tools.debug_utils` modulon keresztül.
