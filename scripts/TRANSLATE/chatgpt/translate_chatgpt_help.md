# translate_chatgpt – ChatGPT alapú JSON fordítás
**Runtime:** `sync`  
**Entry point:** `TRANSLATE/chatgpt/translate_chatgpt.py`

## Overview
Ez a modul a projekt `config.json` beállításai alapján megtalálja a szétválasztott beszéd szegmenseket tartalmazó JSON fájlt, majd a ChatGPT API segítségével lefordítja a hiányzó szövegeket. A feldolgozás intelligens csoportokra bontva, hibák esetén rekurzív felosztással történik, és folytatható a progress fájl újrafelhasználásával. A script a kimeneti JSON-ban metaadatként elmenti a használt system promptot, modellt és nyelvpárt.

## Required Parameters
- `-p/--project-name`: A workdir-en belüli projektmappa neve.

## Optional Parameters
- `-auth_key/--auth-key`: OpenAI API kulcs. Megadáskor elmentésre kerül a `keyholder.json` fájlba. Alapértelmezés: `null`.
- `-input_language/--input-language`: Bemeneti nyelv kódja. Ha nincs megadva, a `config.json` `default_source_lang` értéke használódik.
- `-output_language/--output-language`: Kimeneti nyelv kódja. Ha nincs megadva, a `config.json` `default_target_lang` értéke használódik.
- `-context/--context`: Rövid kontextus, ami bekerül a system promptba. Alapértelmezés: `null`.
- `-model/--model`: Használt OpenAI modell neve. Alapértelmezés: `gpt-4o`.
- `-stream/--stream`: Ha megadod, a script soronként kiírja a fordítási előrehaladást. Alapértelmezés: `false`.
- `-allow_sensitive_content/--allow-sensitive-content`: Speciális system promptot engedélyez kényes tartalmakhoz. Alapértelmezés: `false`.
- `-systemprompt/--systemprompt`: Egyedi system prompt szöveg. Alapértelmezés: `"You are an expert translator. Translate the numbered list from the source language to the target language. Your response MUST be a numbered list with the exact same number of items. Format: \`1. [translation]\`."`
- `--debug`: Debug mód, amely részletesebb naplózást ad. Alapértelmezés: `false`.

-## Outputs
- Frissített JSON a projekt `4_translated_json` könyvtárában, kitöltött `translated_text` mezőkkel és `metadata.translation_*` metaadatokkal.
- Ideiglenes `*.progress.json` fájl ugyanebben a könyvtárban, amely a futás végén automatikusan törlődik.
- A mentett JSON automatikusan átfut a `tools/json_sanitizer.py` tisztító lépésén.

## Error Handling / Tips
- Győződj meg róla, hogy a `keyholder.json` elérhető és a `chatgpt_api_key` mezőt tartalmazza, különben add meg az API kulcsot a CLI-ben.
- A script hibával kilép, ha nem találja a bemeneti JSON-t vagy ha a `config.json` hiányos; futás előtt ellenőrizd a projekt struktúrát.
- Hálózati vagy modellhibák esetén a script a batch-et felosztja és újrapróbálja, de tartós hiba esetén megszakítja a futást és meghagyja a progress fájlt folytatáshoz.
