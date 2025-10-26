# e2e_diarize_speech – konfigurációs útmutató

**Futtatási környezet:** `nemo`  
**Belépési pont:** `e2e_diarize_speech.py`

Figyelem: a JSON is jelzi, hogy ez egy alfa állapotú eszköz. A szkript a NeMo Sortformer-alapú end-to-end diarizációs modelleket futtatja Hydra-konfigurációval, opcionális Optuna utófeldolgozás-optimalizálással. A paramétereket `kulcs=érték` formában kell átadni.

## Kötelező beállítások
- `model_path` (`config_option`, alapértelmezés: nincs): A .nemo vagy .ckpt modellfájl elérési útja, amelyet inferenciára használ.
- `dataset_manifest` (`config_option`, alapértelmezés: nincs): A diabemenet manifest JSON, amely tartalmazza a feldolgozandó audiókra mutató útvonalakat (és opcionálisan RTTM fájlokat evaluációhoz).

## Opcionális beállítások
- `presort_manifest` (alapértelmezés: `true`): Ha igaz, a bemeneti manifestet rendezve tölti be, ami gyorsíthatja a feldolgozást.
- `postprocessing_yaml` (alapértelmezés: nincs): Külső YAML fájl a VAD utófeldolgozási paramétereivel. Ha megadod, a szkript ennek megfelelően állítja be a binarizálást.
- `no_der` (alapértelmezés: `false`): Igazra állítva kihagyja a DER (Diarization Error Rate) kiszámítását.
- `out_rttm_dir` (alapértelmezés: nincs): Mappa, ahová a generált RTTM-eket menti. Üresen hagyva nem készül RTTM kimenet.
- `save_preds_tensors` (alapértelmezés: `false`): Bekapcsolva .pt fájlokba menti a modell nyers kimeneti tenzorait későbbi elemzéshez.
- `session_len_sec` (alapértelmezés: `-1`): A streaming diarizáció ablakának hossza másodpercben. `-1` esetén a lehető leghosszabb szakaszokat használja.
- `batch_size` (alapértelmezés: `1`): Batchméret inferenciához. Az 1-es érték biztosítja a leghosszabb, legpontosabb ablakot.
- `num_workers` (alapértelmezés: `0`): PyTorch DataLoader dolgozók száma; nagyobb érték párhuzamosítja a betöltést.
- `random_seed` (alapértelmezés: nincs): Fixálja a random magot (`seed_everything`) reprodukálhatóság érdekében.
- `bypass_postprocessing` (alapértelmezés: `true`): Igaz esetén kihagyja a VAD utófeldolgozást, és csak a binarizált eredményt adja vissza.
- `log` (alapértelmezés: `false`): Naplózást kapcsol be a NeMo/Lightning komponensek számára.
- `use_lhotse` (alapértelmezés: `true`): Ha hamis, a Lhotse-alapú előfeldolgozás letiltásra kerül.
- `batch_duration` (alapértelmezés: `33000`): A streaming feldolgozásnál használt keretszám (sokszor mintaszám). Alapértelmezett értéke a hivatalos példákból származik.
- `collar` (alapértelmezés: `0.25`): DER számításnál alkalmazott collar (másodpercben), amely tolerálja a szegmenshatárok körüli eltérést.
- `ignore_overlap` (alapértelmezés: `false`): Igazra állítva a DER csak a nem átfedő beszédszakaszokra számolódik.
- `spkcache_len` (alapértelmezés: `188`): A streaming speaker cache hossza keretben, amely meghatározza, mennyi múltbéli információt tart fenn a modell.
- `spkcache_refresh_rate` (alapértelmezés: `144`): Hány keretenként frissül a speaker cache.
- `fifo_len` (alapértelmezés: `188`): A streaming FIFO hossz, amely a szegmens-buffer méretét határozza meg.
- `chunk_len` (alapértelmezés: `6`): Streaming darabok hossza másodpercben.
- `chunk_left_context` (alapértelmezés: `1`): Bal oldali kontextus chunkok száma a streaming feldolgozáshoz.
- `chunk_right_context` (alapértelmezés: `7`): Jobb oldali kontextus chunkok száma.
- `cuda` (alapértelmezés: nincs): GPU index; negatív vagy üres érték esetén CPU-n futtatja a modellt.
- `matmul_precision` (alapértelmezés: `highest`): PyTorch mátrixszorzás pontossága (`highest`, `high`, `medium`) a teljesítmény és memória igény szabályozására.
- `launch_pp_optim` (alapértelmezés: `false`): Igazra állítva Optuna keresést indít a postprocessing paraméterek finomhangolására.
- `optuna_study_name` (alapértelmezés: `optim_postprocessing`): Az Optuna „study” neve; a kapcsolódó fájlok is ezt fogják tartalmazni.
- `optuna_temp_dir` (alapértelmezés: `/tmp/optuna`): Ide kerülnek az ideiglenes fájlok Optuna futtatásakor.
- `optuna_storage` (alapértelmezés: `sqlite:///optim_postprocessing.db`): Az Optuna perzisztens tárolójának URI-ja.
- `optuna_log_file` (alapértelmezés: `optim_postprocessing.log`): Logfájl név, amelybe az Optuna futás naplóz.
- `optuna_n_trials` (alapértelmezés: `100000`): Futási próbák felső korlátja Optuna optimalizáció során; nagy érték mély keresést enged, de hosszabb ideig tart.
