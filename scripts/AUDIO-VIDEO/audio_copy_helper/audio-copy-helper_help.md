# audio-copy-helper – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `audio-copy-helper.py`

A segédszkript a projekt `upload` mappájában lévő audió fájlokat 44,1 kHz-es WAV formátumba konvertálja, majd a megadott célmappákba másolja őket. A háttér sáv esetében néma (0 hangerős) változatot készít.

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): Annak a projektmappának a neve, amelynek `upload` könyvtárában találhatók az átkonvertálandó fájlok.

## Opcionális beállítások
- `extracted_audio` (`--extracted_audio`, flag, alapértelmezés: `false`): A konvertált fájlt elhelyezi a `1.5_extracted_audio` mappában.
- `separated_audio_background` (`--separated_audio_background`, flag, alapértelmezés: `false`): Néma másolatot készít a `2_separated_audio_background` mappába.
- `separated_audio_speech` (`--separated_audio_speech`, flag, alapértelmezés: `false`): A konvertált hangot a `2_separated_audio_speech` mappába másolja.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletesebb naplózás engedélyezése.

> 💡 Tipp: legalább egy célkapcsolót aktiválj, különben a szkript hibaüzenettel leáll.
