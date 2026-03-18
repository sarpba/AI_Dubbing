# canary-easy

**Futtatási környezet:** `nemo`  
**Belépési pont:** `ASR/canary/canary-easy.py`

## Mit csinál?
A Canary ASR modell használata a beszéd szöveggé alakítására.

A script a projekt hanganyagából készít átírást vagy újraszegmentált JSON-t, hogy a további fordítási és TTS lépések már strukturált bemenettel dolgozzanak.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.
- `source_lang` (opció;  kapcsoló: `--source-lang`; alapértelmezés: üres érték): A bemeneti beszéd nyelve. Többnyelvű modelleknél ez segíti a pontosabb felismerést.
- `target_lang` (opció;  kapcsoló: `--target-lang`; alapértelmezés: üres érték): A cél nyelv. Fordító vagy speech-to-speech modelleknél ez határozza meg a kimenet nyelvét.

## Opcionális paraméterek
- `model_name` (opció;  kapcsoló: `--model-name`; alapértelmezés: `nvidia/canary-1b-v2`): A használt ASR modell neve vagy azonosítója.
- `batch_size` (opció;  kapcsoló: `--batch-size`; alapértelmezés: `4`): Az egyszerre feldolgozott elemek száma. Nagyobb érték gyorsabb lehet, de több memóriát igényel.
- `beam_size` (opció;  kapcsoló: `--beam-size`; alapértelmezés: `5`): A keresés szélessége. Nagyobb érték jobb minőséget, de lassabb futást eredményezhet.
- `len_pen` (opció;  kapcsoló: `--len-pen`; alapértelmezés: `1.0`): Hossz-büntetés a dekódolásnál. A kimenet hosszát befolyásolja.
- `chunk` (opció;  kapcsoló: `--chunk`; alapértelmezés: `30`): A feldolgozásnál használt darabolás mérete másodpercben.
- `max_pause` (opció;  kapcsoló: `--max-pause`; alapértelmezés: `0.6`): Legfeljebb ekkora szünetet hagy a script egy szegmensen belül. Nagyobb érték hosszabb mondatrészeket eredményezhet.
- `timestamp_padding` (opció;  kapcsoló: `--timestamp-padding`; alapértelmezés: `0.2`): Ennyi időt ad hozzá a szegmensek elejéhez és végéhez a kényelmesebb vágás érdekében.
- `max_segment_duration` (opció;  kapcsoló: `--max-segment-duration`; alapértelmezés: `11.5`): A szegmensek maximális hossza másodpercben.
- `keep_alternatives` (opció;  kapcsoló: `--keep-alternatives`; alapértelmezés: `2`): Ennyi alternatív felismerési lehetőséget tart meg, ha a script támogatja.
- `overwrite` (kapcsoló;  kapcsoló: `--overwrite`; alapértelmezés: `false`): A már meglévő kimeneti fájlokat is újragenerálja vagy felülírja. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
