# vibevoice-tts

**Futtatási környezet:** `vibevoice`  
**Belépési pont:** `TTS/vibevoice_old/vibevoice-tts.py`

## Mit csinál?
A VibeVoice modellt és opcionális LoRA adaptereket használva hozza létre a szinkron szegmenseket a fordított állományból.

A script a lefordított szegmensekből generál szinkronhangot, és a létrejött hangfájlokat a projekt TTS kimenetei közé menti.

## Kötelező paraméterek
- `project_name` (pozicionális;  kapcsoló: pozicionális; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.
- `norm` (opció;  kapcsoló: `--norm`; alapértelmezés: nincs): A használt normalizálási profil neve. Ez határozza meg a szöveg-előkészítés nyelvi szabályait.

## Opcionális paraméterek
- `model_path` (opció;  kapcsoló: `--model_path`; alapértelmezés: `microsoft/VibeVoice-1.5b`): A betöltendő modell Hugging Face azonosítója vagy lokális útvonala.
- `model_dir` (opció;  kapcsoló: `--model_dir`; alapértelmezés: nincs): Lokális modellkönyvtár. Ha meg van adva, a script ezt részesíti előnyben a távoli modellazonosítóval szemben.
- `checkpoint_path` (opció;  kapcsoló: `--checkpoint_path`; alapértelmezés: nincs): Opcionális checkpoint vagy adapter fájl, amelyet a script az alapmodell fölé tölt be.
- `device` (opció;  kapcsoló: `--device`; alapértelmezés: `auto`): A futtatás eszköze, például `cpu`, `cuda`, `cuda:0` vagy `mps`.
- `cfg_scale` (opció;  kapcsoló: `--cfg_scale`; alapértelmezés: `1.3`): A stílus- és tartalomkövetés erőssége generálás közben.
- `disable_prefill` (kapcsoló;  kapcsoló: `--disable_prefill`; alapértelmezés: `false`): Kikapcsolja az előfeltöltési vagy voice-cloning előkészítő lépést. Alapállapotban ki van kapcsolva.
- `ddpm_steps` (opció;  kapcsoló: `--ddpm_steps`; alapértelmezés: `10`): A diffúziós generálás lépésszáma.
- `eq_config` (opció;  kapcsoló: `--eq_config`; alapértelmezés: nincs): EQ beállításokat tartalmazó JSON fájl a referenciahang előkészítéséhez.
- `normalize_ref_audio` (kapcsoló;  kapcsoló: `--normalize_ref_audio`; alapértelmezés: `false`): A referenciahangot egységes hangerőre normalizálja a generálás előtt. Alapállapotban ki van kapcsolva.
- `ref_audio_peak` (opció;  kapcsoló: `--ref_audio_peak`; alapértelmezés: `0.95`): A referenciahang cél csúcsszintje normalizáláskor.
- `target_sample_rate` (opció;  kapcsoló: `--target_sample_rate`; alapértelmezés: `16000`): A referenciahang cél mintavételi frekvenciája.
- `speaker_name` (opció;  kapcsoló: `--speaker_name`; alapértelmezés: `Speaker 1`): A generált szöveghez és hanghoz használt beszélőnév vagy címke.
- `max_retries` (opció;  kapcsoló: `--max_retries`; alapértelmezés: `5`): Ennyiszer próbálkozik újra, ha az ellenőrzés vagy a generálás nem ad elfogadható eredményt.
- `tolerance_factor` (opció;  kapcsoló: `--tolerance_factor`; alapértelmezés: `1.0`): Az ellenőrző összehasonlítás tűrésének szorzója.
- `min_tolerance` (opció;  kapcsoló: `--min_tolerance`; alapértelmezés: `2`): A minimálisan megengedett tűrés.
- `whisper_model` (opció;  kapcsoló: `--whisper_model`; alapértelmezés: `openai/whisper-large-v3`): A belső ellenőrzéshez használt Whisper modell.
- `beam_size` (opció;  kapcsoló: `--beam_size`; alapértelmezés: `5`): A keresés szélessége. Nagyobb érték jobb minőséget, de lassabb futást eredményezhet.
- `seed` (opció;  kapcsoló: `--seed`; alapértelmezés: `-1`): A véletlenmag. Fix értékkel reprodukálhatóbb, `-1` esetén változatosabb eredmény várható.
- `max_segments` (opció;  kapcsoló: `--max_segments`; alapértelmezés: nincs): A feldolgozható szegmensek számát korlátozza teszteléshez vagy gyors próbafutáshoz.
- `overwrite` (kapcsoló;  kapcsoló: `--overwrite`; alapértelmezés: `false`): A már meglévő kimeneti fájlokat is újragenerálja vagy felülírja. Alapállapotban ki van kapcsolva.
- `save_failures` (kapcsoló;  kapcsoló: `--save_failures`; alapértelmezés: `false`): Elmenti a sikertelen vagy elutasított generálásokat további elemzésre. Alapállapotban ki van kapcsolva.
- `keep_best_over_tolerance` (kapcsoló;  kapcsoló: `--keep_best_over_tolerance`; alapértelmezés: `false`): Ha nincs teljesen megfelelő eredmény, megtartja a legjobb próbálkozást is. Alapállapotban ki van kapcsolva.
- `max_workers` (opció;  kapcsoló: `--max_workers`; alapértelmezés: nincs): A párhuzamosan dolgozó munkafolyamatok maximális száma.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
