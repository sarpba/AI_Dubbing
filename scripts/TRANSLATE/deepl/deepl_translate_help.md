# deepl_translate

**Futtatási környezet:** `sync`  
**Belépési pont:** `TRANSLATE/deepl/deepl_translate.py`

## Mit csinál?
Időzített transcripciók fordítása különböző fordító szolgáltatások segítségével. Nyersfordítás, nem támaszkodik feliratokra.

A script a projekt szöveges JSON-jait fordítja a megadott cél nyelvre, és a fordítást a projekt megfelelő kimeneti mappájába menti.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `auth_key` (opció;  kapcsoló: `-auth_key`, `--auth-key`; alapértelmezés: nincs): API kulcs az adott külső szolgáltatáshoz. Megadva a rendszer elmentheti későbbi használatra.
- `input_language` (opció;  kapcsoló: `-input_language`, `--input-language`; alapértelmezés: nincs): A forrás szöveg nyelve. Ha nincs megadva, a script a konfigurációból vagy automatikus felismerésből indul ki.
- `output_language` (opció;  kapcsoló: `-output_language`, `--output-language`; alapértelmezés: nincs): A lefordított szöveg cél nyelve.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
