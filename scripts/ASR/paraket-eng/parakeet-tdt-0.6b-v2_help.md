# parakeet-tdt-0.6b-v2

**Futtatási környezet:** `nemo`  
**Belépési pont:** `ASR/paraket-eng/parakeet-tdt-0.6b-v2.py`

## Mit csinál?
A Parakeet TDT 0.6B V2 modell használata a beszéd szöveggé alakítására.

A script a projekt hanganyagából készít átírást vagy újraszegmentált JSON-t, hogy a további fordítási és TTS lépések már strukturált bemenettel dolgozzanak.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `no_auto_chunk` (kapcsoló;  kapcsoló: `--no-auto-chunk`; alapértelmezés: `false`): A `no auto chunk` funkció kapcsolója. Alapállapotban ki van kapcsolva; a kapcsoló aktiválásakor a script letilt egy belső funkciót.
- `chunk` (opció;  kapcsoló: `--chunk`; alapértelmezés: `30`): A feldolgozásnál használt darabolás mérete másodpercben.
- `max_pause` (opció;  kapcsoló: `--max-pause`; alapértelmezés: `0.6`): Legfeljebb ekkora szünetet hagy a script egy szegmensen belül. Nagyobb érték hosszabb mondatrészeket eredményezhet.
- `timestamp_padding` (opció;  kapcsoló: `--timestamp-padding`; alapértelmezés: `0.2`): Ennyi időt ad hozzá a szegmensek elejéhez és végéhez a kényelmesebb vágás érdekében.
- `max_segment_duration` (opció;  kapcsoló: `--max-segment-duration`; alapértelmezés: `11.5`): A szegmensek maximális hossza másodpercben.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
