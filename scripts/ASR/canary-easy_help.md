# canary-easy – konfigurációs útmutató

**Futtatási környezet:** `parakeet`  
**Belépési pont:** `ASR/canary-easy.py`

A szkript az NVIDIA Canary ASR modellt futtatja a `separated_audio_speech` mappában található hangfájlokon. A hosszabb felvételeket automatikusan darabolja, majd JSON formátumban menti a szöveget, időbélyegeket és – igény szerint – alternatív hipotéziseket.

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatt található projektmappa neve, amelynek hanganyagát fel kell dolgozni.

## Opcionális beállítások
- `model_name` (`--model-name`, option, alapértelmezés: `nvidia/canary-1b-v2`): A használandó Canary modell huggingface azonosítója. Más modell megadásával eltérő nyelvi/szolgáltatási profilt választhatsz.
- `batch_size` (`--batch-size`, option, alapértelmezés: `4`): Hány hangminta kerüljön egyszerre GPU-ra a dekódolás során. Nagyobb érték gyorsíthat, de több memóriát kér.
- `beam_size` (`--beam-size`, option, alapértelmezés: `5`): A beam-search szélessége. Nagyobb érték javíthatja a pontosságot, viszont lassabbá teszi a dekódolást.
- `len_pen` (`--len-pen`, option, alapértelmezés: `1.0`): Hossz-büntetés a dekóderben; 1 feletti érték a rövidebb, 1 alatti a hosszabb hipotéziseket preferálja.
- `chunk` (`--chunk`, option, alapértelmezés: `30`): Chunkhossz másodpercben, ha manuálisan akarod szabályozni a feldarabolást (10–120 mp között célszerű).
- `source_lang` (`--source-lang`, option, alapértelmezés: `auto`): Forrásnyelv kódja. `auto` esetén a Canary próbálja felismerni.
- `target_lang` (`--target-lang`, option, alapértelmezés: nincs): Cél nyelv kódja, ha a Canary fordítást is végezzen. Üresen hagyva csak átírás készül.
- `keep_alternatives` (`--keep-alternatives`, option, alapértelmezés: `2`): Chunkonként ennyi alternatív hipotézist tárol a kimenő JSON-ban.
- `overwrite` (`--overwrite`, flag, alapértelmezés: `false`): Ha engedélyezed, felülírja a már létező kimeneti fájlokat; ellenkező esetben kihagyja őket.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás a `tools.debug_utils` segítségével.
