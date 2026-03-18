# collect_normalized_translations

**Futtatási környezet:** `f5-tts`  
**Belépési pont:** `NORMALIZER_HELPER/collect_normalized_translations.py`

## Mit csinál?
Az idegen nyelvű szavakat kigyüjti a normalizáló canges_new.csv fileba.

A script a projekt fordításaiból kigyűjti azokat az elemeket, amelyek hasznosak lehetnek a normalizáló CSV-k bővítéséhez.

## Kötelező paraméterek
- `project_dir` (opció;  kapcsoló: `-p`, `--project`; alapértelmezés: nincs): A projekt könyvtára, amelynek fordításaiból a script kigyűjti az új normalizálandó elemeket.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
