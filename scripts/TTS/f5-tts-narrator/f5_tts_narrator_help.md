# f5_tts_narrator

**Futtatási környezet:** `f5-tts`  
**Belépési pont:** `TTS/f5-tts-narrator/f5_tts_narrator.py`

## Mit csinál?
Narrátor referencia alapján generálja az összes szinkron szegmenst F5-TTS-sel.

A script a lefordított szegmensekből generál szinkronhangot, és a létrejött hangfájlokat a projekt TTS kimenetei közé menti.

## Kötelező paraméterek
- `project_name` (pozicionális;  kapcsoló: pozicionális; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.
- `norm` (opció;  kapcsoló: `--norm`; alapértelmezés: nincs): A használt normalizálási profil neve. Ez határozza meg a szöveg-előkészítés nyelvi szabályait.
- `narrator` (opció;  kapcsoló: `--narrator`; alapértelmezés: nincs): A narrátor referenciahangját tartalmazó fájl vagy mappa.

## Opcionális paraméterek
- `model_dir` (opció;  kapcsoló: `--model_dir`; alapértelmezés: nincs): Lokális modellkönyvtár. Ha meg van adva, a script ezt részesíti előnyben a távoli modellazonosítóval szemben.
- `speed` (opció;  kapcsoló: `--speed`; alapértelmezés: `1.0`): A generált beszéd sebessége.
- `nfe_step` (opció;  kapcsoló: `--nfe_step`; alapértelmezés: `32`): Az F5-TTS mintavételi lépésszáma.
- `remove_silence` (kapcsoló;  kapcsoló: `--remove_silence`; alapértelmezés: `false`): Eltávolítja a felesleges csendet a generált kimenetből. Alapállapotban ki van kapcsolva.
- `phonetic_ref` (kapcsoló;  kapcsoló: `--phonetic-ref`; alapértelmezés: `false`): A referenciafeldolgozásnál fonetikus megközelítést használ. Alapállapotban ki van kapcsolva.
- `normalize_ref_audio` (kapcsoló;  kapcsoló: `--normalize-ref-audio`; alapértelmezés: `false`): A referenciahangot egységes hangerőre normalizálja a generálás előtt. Alapállapotban ki van kapcsolva.
- `eq_config` (opció;  kapcsoló: `--eq-config`; alapértelmezés: nincs): EQ beállításokat tartalmazó JSON fájl a referenciahang előkészítéséhez.
- `ref_audio_peak` (opció;  kapcsoló: `--ref-audio-peak`; alapértelmezés: `0.95`): A referenciahang cél csúcsszintje normalizáláskor.
- `max_workers` (opció;  kapcsoló: `--max_workers`; alapértelmezés: nincs): A párhuzamosan dolgozó munkafolyamatok maximális száma.
- `seed` (opció;  kapcsoló: `--seed`; alapértelmezés: `-1`): A véletlenmag. Fix értékkel reprodukálhatóbb, `-1` esetén változatosabb eredmény várható.
- `max_retries` (opció;  kapcsoló: `--max-retries`; alapértelmezés: `5`): Ennyiszer próbálkozik újra, ha az ellenőrzés vagy a generálás nem ad elfogadható eredményt.
- `tolerance_factor` (opció;  kapcsoló: `--tolerance-factor`; alapértelmezés: `1.0`): Az ellenőrző összehasonlítás tűrésének szorzója.
- `min_tolerance` (opció;  kapcsoló: `--min-tolerance`; alapértelmezés: `2`): A minimálisan megengedett tűrés.
- `whisper_model` (opció;  kapcsoló: `--whisper-model`; alapértelmezés: `openai/whisper-large-v3`): A belső ellenőrzéshez használt Whisper modell.
- `beam_size` (opció;  kapcsoló: `--beam-size`; alapértelmezés: `5`): A keresés szélessége. Nagyobb érték jobb minőséget, de lassabb futást eredményezhet.
- `save_failures` (kapcsoló;  kapcsoló: `--save-failures`; alapértelmezés: `false`): Elmenti a sikertelen vagy elutasított generálásokat további elemzésre. Alapállapotban ki van kapcsolva.
- `keep_best_over_tolerance` (kapcsoló;  kapcsoló: `--keep-best-over-tolerance`; alapértelmezés: `false`): Ha nincs teljesen megfelelő eredmény, megtartja a legjobb próbálkozást is. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
