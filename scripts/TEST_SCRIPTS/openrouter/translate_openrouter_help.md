# translate_openrouter (test) – OpenRouter alapú JSON fordítás
**Runtime:** `sync`  
**Entry point:** `TEST_SCRIPTS/openrouter/translate_openrouter.py`

## Overview
Ez a teszt-script a projekt `config.json` beállításai alapján feltérképezi a szétválasztott beszédszegmenseket tartalmazó JSON-t, majd OpenRouteren keresztül hívott modellekkel fordítja le a hiányzó részeket. A feldolgozás intelligens csoportokra bontva, rekurzív újrapróbálással, részhaladási fájl támogatással és API-naplózással történik. A kimeneti JSON metaadatai tartalmazzák a promptot, modellt, nyelvpárt és a szolgáltató nevét.

## Required Parameters
- `-p/--project-name`: A workdir-en belüli projektmappa neve.

## Optional Parameters
- `-auth_key/--auth-key`: OpenRouter API kulcs. Megadáskor elmentésre kerül a `keyholder.json` fájlba. Alapértelmezés: `null`.
- `-input_language/--input-language`: Bemeneti nyelv kódja. Ha hiányzik, a `config.json` `default_source_lang` értéke lép életbe.
- `-output_language/--output-language`: Kimeneti nyelv kódja. Ha hiányzik, a `config.json` `default_target_lang` értéke lép életbe.
- `-context/--context`: Rövid kontextus, ami bekerül a promptba. Alapértelmezés: `null`.
- `--tone`: Elvárt hangnem/stílus (pl. „szarkasztikus, laza”). A prompt hangsúlyozza ezt.
- `--target-audience`: Célközönség megadása (pl. „magyar Netflix nézők”), hogy a modell ennek megfelelő nyelvezetet használjon.
- `--platform`: Platform vagy formátum (pl. Netflix, YouTube, podcast). Meghatározza a sorhossz/stílus elvárásokat.
- `--style-notes`: További szabad formájú stílusinstrukciók, amelyek szó szerint bekerülnek a promptba.
- `--glossary`: Opcionális JSON fájl (forrás→cél kulcsszópárok) a következetes terminológiához. Relatív útvonalat is megadhatsz a projektgyökérhez képest.
- `-model/--model`: Használt OpenRouter modell neve. Alapértelmezés: `google/gemini-2.0-flash-001`.
- `-stream/--stream`: Ha megadod, a script soronként logolja a lefordított sorokat. Alapértelmezés: `false`.
- `-allow_sensitive_content/--allow-sensitive-content`: Engedélyezi a kényes tartalmak explicit fordítását is kérő promptot. Alapértelmezés: `false`.
- `-systemprompt/--systemprompt`: Egyedi system prompt. Ha megadod, felülírja a script által generált, feliratfordításra optimalizált instrukciókat (támogatja a `{source_language}`, `{target_language}`, `{source_language_code}`, `{target_language_code}` helyőrzőket).
- `--debug`: Debug mód, részletes naplózással. Alapértelmezés: `false`.

## Outputs
- Frissített JSON a projekt `4_translated_json` könyvtárában, kitöltött `translated_text` mezőkkel, `metadata.translation_*` értékekkel és `metadata.translation_provider = "openrouter"` mezővel.
- Ideiglenes `*.progress.json` fájl ugyanebben a könyvtárban; sikeres futás végén törlődik, hiba esetén folytatáshoz használható.
- API-hívás napló (`openrouter_api_YYYYMMDD_HHMMSS.jsonl`) a projekt log könyvtárában.
- A mentett JSON a folyamat végén automatikusan átfut a `tools/json_sanitizer.py` tisztítóján.

## Error Handling / Tips
- Győződj meg róla, hogy a `keyholder.json` elérhető és tartalmazza az `openrouter_api_key` mezőt; ha nem, add meg a kulcsot CLI paraméterként.
- Ha rendelkezel publikus URL-lel vagy alkalmazásnévvel, állítsd be az `OPENROUTER_HTTP_REFERER` és `OPENROUTER_APP_TITLE` környezeti változókat az OpenRouter irányelveihez igazodva.
- A beépített prompt automatikusan igazodik a forrásnyelvhez, explicit szerepet, stílust, platformot, szlengkezelést és glosszárium emlékeztetőt tartalmaz. Egyedi `-systemprompt` megadásakor minden automatikus stratégia felülíródik.
- Glosszárium használatakor ügyelj a helyes JSON formátumra; ha hibás, a script figyelmeztet és glosszárium nélkül fut tovább.
- Hálózati vagy modellhibák esetén a script a batch-et felosztja és újrapróbálja, de tartós hiba esetén meghagyja a progress fájlt, így később folytatható.
