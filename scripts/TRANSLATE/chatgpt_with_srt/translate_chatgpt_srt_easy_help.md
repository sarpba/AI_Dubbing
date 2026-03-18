# translate_chatgpt_srt_easy

**Futtatási környezet:** `sync`  
**Belépési pont:** `TRANSLATE/chatgpt_with_srt/translate_chatgpt_srt_easy.py`

## Mit csinál?
A beolvasott időzített transcripciót a rendelkezésre álló feliratot felhasználva lefordítja a kívánt nyelvre a ChatGPT segítségével.

A script a projekt szöveges JSON-jait fordítja a megadott cél nyelvre, és a fordítást a projekt megfelelő kimeneti mappájába menti.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-project_name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `input_language` (opció;  kapcsoló: `-input_language`; alapértelmezés: `EN`): A forrás szöveg nyelve. Ha nincs megadva, a script a konfigurációból vagy automatikus felismerésből indul ki.
- `output_language` (opció;  kapcsoló: `-output_language`; alapértelmezés: `HU`): A lefordított szöveg cél nyelve.
- `auth_key` (opció;  kapcsoló: `-auth_key`; alapértelmezés: nincs): API kulcs az adott külső szolgáltatáshoz. Megadva a rendszer elmentheti későbbi használatra.
- `context` (opció;  kapcsoló: `-context`; alapértelmezés: nincs): Rövid tartalmi kontextus, ami segíti a modell stílus- és szóhasználatbeli döntéseit.
- `model` (opció;  kapcsoló: `-model`; alapértelmezés: `gpt-4o`): A használt modell neve vagy azonosítója.
- `stream` (kapcsoló;  kapcsoló: `-stream`; alapértelmezés: `false`): A válasz és az előrehaladás folyamatos kiírása futás közben. Alapállapotban ki van kapcsolva.
- `allow_sensitive_content` (kapcsoló;  kapcsoló: `--allow-sensitive-content`, `--no-allow-sensitive-content`; alapértelmezés: `true`): Engedélyezi, hogy a fordítás érzékeny, nyers vagy felnőtt tartalmat is pontosabban kezeljen. Mindkét állapot explicit megadható a kapcsolópárral.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
