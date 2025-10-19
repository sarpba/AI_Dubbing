# translate_chatgpt_srt_easy_codex – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**API:** `chatgpt`  
**Belépési pont:** `TRANSLATE/translate_chatgpt_srt_easy_codex.py`

A szkript a ChatGPT API-t használja arra, hogy a projekt időzített transzkripcióját cél nyelvre fordítsa, miközben opcionálisan figyelembe veszi az eredeti feliratot (SRT) kontextusként. A kulcsokat a `keyholder.json` fájlban tárolja, és gondoskodik a chunkolt feldolgozásról a tokénlimit elkerülésére.

## Kötelező beállítás
- `project_name` (`-project_name`, option, alapértelmezés: nincs): A `workdir` alatti projekt neve, amelynek anyagát fordítani kell.

## Opcionális beállítások
- `input_language` (`-input_language`, option, alapértelmezés: `EN`): A transzkriptben várható bemeneti nyelv kódja. A script ezt használja a kontextusfájl kiválasztásához.
- `output_language` (`-output_language`, option, alapértelmezés: `HU`): A kívánt célnyelv kódja; az SRT kontextuskeresés is ezt veszi alapul.
- `auth_key` (`-auth_key`, option, alapértelmezés: nincs): OpenAI API kulcs. Megadáskor base64 kódolással a `keyholder.json`-ba menti, különben onnan próbálja visszatölteni.
- `context` (`-context`, option, alapértelmezés: nincs): A paraméter jelen változatban figyelmen kívül marad; megtartották kompatibilitási okokból.
- `model` (`-model`, option, alapértelmezés: `gpt-4o`): A használandó OpenAI modell azonosítója. Bármely támogatott ChatGPT modell megadható.
- `stream` (`-stream`, flag, alapértelmezés: `false`): Valós idejű streamelt választ kér a kliensből, így futás közben is látszik az előrehaladás.
- `allow_sensitive_content` (`--allow-sensitive-content`/`--no-allow-sensitive-content`, flag, alapértelmezés: `true`): A jelenlegi implementáció nem használja, de a CLI kompatibilitás miatt szerepel.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Extra naplózást kapcsol be a `tools.debug_utils` modulon keresztül.
