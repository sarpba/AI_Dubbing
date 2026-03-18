# resegment

**Futtatási környezet:** `sync`  
**Belépési pont:** `ASR/resegment/resegment.py`

## Mit csinál?
JSON szegmensek újraformázása - ASR script által létrehozott JSON fájlok biztonsági mentése és újraformázása különböző paraméterekkel.

A script a projekt hanganyagából készít átírást vagy újraszegmentált JSON-t, hogy a további fordítási és TTS lépések már strukturált bemenettel dolgozzanak.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `max_pause` (opció;  kapcsoló: `--max-pause`; alapértelmezés: `0.8`): Legfeljebb ekkora szünetet hagy a script egy szegmensen belül. Nagyobb érték hosszabb mondatrészeket eredményezhet.
- `timestamp_padding` (opció;  kapcsoló: `--timestamp-padding`; alapértelmezés: `0.1`): Ennyi időt ad hozzá a szegmensek elejéhez és végéhez a kényelmesebb vágás érdekében.
- `max_segment_duration` (opció;  kapcsoló: `--max-segment-duration`; alapértelmezés: `11.5`): A szegmensek maximális hossza másodpercben.
- `enforce_single_speaker` (kapcsoló;  kapcsoló: `--enforce-single-speaker`; alapértelmezés: `false`): Arra kényszeríti az újraszegmentálást, hogy egy szegmens lehetőleg csak egy beszélőt tartalmazzon. Alapállapotban ki van kapcsolva.
- `backup` (kapcsoló;  kapcsoló: `--backup`; alapértelmezés: `true`): A script biztonsági mentést készít a módosított fájlokról a feldolgozás előtt. Alapállapotban be van kapcsolva.
- `no_backup` (kapcsoló;  kapcsoló: `--no-backup`; alapértelmezés: `false`): Kikapcsolja a biztonsági mentés készítését. Alapállapotban a script mentést készít a módosított fájlokról. Alapállapotban ki van kapcsolva; a kapcsoló aktiválásakor a script letilt egy belső funkciót.
- `skip_energy_refine` (kapcsoló;  kapcsoló: `--skip-energy-refine`; alapértelmezés: `false`): Kihagyja az energiaszint alapú finomítást, ezért gyorsabb, de kevésbé precíz lehet az illesztés. Alapállapotban ki van kapcsolva.
- `word_by_word_segments` (kapcsoló;  kapcsoló: `--word-by-word-segments`; alapértelmezés: `false`): A kimenetet a lehető legkisebb, szóközeli egységekre bontja. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
