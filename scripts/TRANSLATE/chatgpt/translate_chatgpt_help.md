# translate_chatgpt

**Futtatási környezet:** `sync`  
**Belépési pont:** `TRANSLATE/chatgpt/translate_chatgpt.py`

## Mit csinál?
Projekt-alapú JSON fordítás ChatGPT segítségével, intelligens csoportosítással és folytatható futással.

A script a projekt szöveges JSON-jait fordítja a megadott cél nyelvre, és a fordítást a projekt megfelelő kimeneti mappájába menti.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `auth_key` (opció;  kapcsoló: `-auth_key`, `--auth-key`; alapértelmezés: nincs): API kulcs az adott külső szolgáltatáshoz. Megadva a rendszer elmentheti későbbi használatra.
- `input_language` (opció;  kapcsoló: `-input_language`, `--input-language`; alapértelmezés: nincs): A forrás szöveg nyelve. Ha nincs megadva, a script a konfigurációból vagy automatikus felismerésből indul ki.
- `output_language` (opció;  kapcsoló: `-output_language`, `--output-language`; alapértelmezés: nincs): A lefordított szöveg cél nyelve.
- `context` (opció;  kapcsoló: `-context`, `--context`; alapértelmezés: nincs): Rövid tartalmi kontextus, ami segíti a modell stílus- és szóhasználatbeli döntéseit.
- `model` (opció;  kapcsoló: `-model`, `--model`; alapértelmezés: `gpt-4o`): A használt modell neve vagy azonosítója.
- `stream` (kapcsoló;  kapcsoló: `-stream`, `--stream`; alapértelmezés: `false`): A válasz és az előrehaladás folyamatos kiírása futás közben. Alapállapotban ki van kapcsolva.
- `allow_sensitive_content` (kapcsoló;  kapcsoló: `-allow_sensitive_content`, `--allow-sensitive-content`; alapértelmezés: `false`): Engedélyezi, hogy a fordítás érzékeny, nyers vagy felnőtt tartalmat is pontosabban kezeljen. Alapállapotban ki van kapcsolva.
- `systemprompt` (opció;  kapcsoló: `-systemprompt`, `--systemprompt`; alapértelmezés: `You are an expert translator. Translate the numbered list from the source language to the target language. Your response MUST be a numbered list with the exact same number of items. Format: `1. [translation]`.`): Egyedi rendszerprompt. Ezzel teljesen testre szabható a fordítómodell viselkedése.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
