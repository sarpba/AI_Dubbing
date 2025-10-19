# translate – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**API:** `deepl`  
**Belépési pont:** `TRANSLATE/translate.py`

A szkript a DeepL fordító API-t használja, hogy a `separated_audio_speech` könyvtárban található időzített JSON-okat célnyelvre fordítsa. A kimenetet a `translated` almappába menti, és kezeli a `keyholder.json` titkos kulcstárolót.

## Kötelező beállítások
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A projekt neve a `workdir` alatt, amelynek anyagát fordítani kell.
- `auth_key` (`-auth_key`, `--auth-key`, option, alapértelmezés: nincs): DeepL REST API kulcs. Megadáskor a szkript elmenti base64 formában a `keyholder.json` fájlba; ha nem adod meg, előbb onnan próbál olvasni. Valamilyen formában, vagy parancssorban, vagy a webfelületen egyszer kötelezőp megadni, egyébként hibával leáll a script, utána már beolvassa a `keyholder.json`-ból.

## Opcionális beállítások
- `input_language` (`-input_language`, `--input-language`, option, alapértelmezés: config vagy nincs): A forrásnyelv kódja. Üresen hagyva a `config.json` `CONFIG.default_source_lang` beállítására esik vissza.
- `output_language` (`-output_language`, `--output-language`, option, alapértelmezés: config vagy nincs): A célnyelv kódja. Üresen hagyva a `config.json` `CONFIG.default_target_lang` értékét használja.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózást kér a `tools.debug_utils` segítségével (ha a futtatókörnyezet támogatja).
