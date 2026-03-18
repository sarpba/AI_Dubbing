# e2e_diarize_speech

**Futtatási környezet:** `nemo`  
**Belépési pont:** `DIARIZE/speaker_diarize_e2e_non_working_beta/e2e_diarize_speech.py`

## Mit csinál?
Alpha verzió egyelóre ne használd! Beszélődiarizáció végrehajtása az End-to-End Diarization modell segítségével.

Ez egy korai, kísérleti e2e diarizációs script. Főleg haladó tesztelésre való; a paraméterek többsége közvetlenül a modell konfigurációját vezérli.

## Kötelező paraméterek
- `model_path` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: nincs): A diarizációs modell vagy konfiguráció útvonala.
- `dataset_manifest` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: nincs): A diarizációhoz használt manifest fájl.

## Opcionális paraméterek
- `presort_manifest` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `true`): A manifest előrendezése futás előtt.
- `postprocessing_yaml` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: nincs): Utófeldolgozási YAML konfiguráció.
- `no_der` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `false`): Kikapcsolja a DER kiértékelést.
- `out_rttm_dir` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: nincs): Az RTTM kimenetek célkönyvtára.
- `save_preds_tensors` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `false`): Elmenti a köztes predikciós tenzorokat is.
- `session_len_sec` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `-1`): A teljes session hossza másodpercben.
- `batch_size` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `1`): Az egyszerre feldolgozott elemek száma. Nagyobb érték gyorsabb lehet, de több memóriát igényel.
- `num_workers` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `0`): A párhuzamos worker folyamatok száma.
- `random_seed` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: nincs): A véletlenmag a reprodukálható futáshoz.
- `bypass_postprocessing` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `true`): Kihagyja az utófeldolgozási lépéseket.
- `log` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `false`): Külön naplózási mód vagy logfájl-kezelés bekapcsolása.
- `use_lhotse` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `true`): Lhotse-alapú adatkezelést használ, ha a script támogatja.
- `batch_duration` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `33000`): A batch célhossza.
- `collar` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `0.25`): A diarizációs kiértékelésnél használt collar érték.
- `ignore_overlap` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `false`): A kiértékelés vagy utófeldolgozás során figyelmen kívül hagyja az átfedő beszédet.
- `spkcache_len` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `188`): A beszélőcache mérete.
- `spkcache_refresh_rate` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `144`): A beszélőcache frissítési gyakorisága.
- `fifo_len` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `188`): A belső FIFO puffer mérete.
- `chunk_len` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `6`): A diarizációs elemzőablak mérete.
- `chunk_left_context` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `1`): A bal oldali kontextus hossza.
- `chunk_right_context` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `7`): A jobb oldali kontextus hossza.
- `cuda` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: nincs): A CUDA-eszköz kiválasztása vagy engedélyezése.
- `matmul_precision` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `highest`): A mátrixműveletek pontossági beállítása.
- `launch_pp_optim` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `false`): Elindítja az utófeldolgozási optimalizálást.
- `optuna_study_name` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `optim_postprocessing`): Az Optuna study neve.
- `optuna_temp_dir` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `/tmp/optuna`): Az Optuna ideiglenes munkakönyvtára.
- `optuna_storage` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `sqlite:///optim_postprocessing.db`): Az Optuna tároló backendje.
- `optuna_log_file` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `optim_postprocessing.log`): Az Optuna naplófájl neve.
- `optuna_n_trials` (konfiguráció;  kapcsoló: pozicionális; alapértelmezés: `100000`): Az optimalizációs próbálkozások maximális száma.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
