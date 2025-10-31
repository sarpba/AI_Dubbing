# f5_tts_easy_codex_EQ – konfigurációs útmutató

**Futtatási környezet:** `f5-tts`  
**Belépési pont:** `f5_tts_easy_EQ.py`

A szkript a finomhangolt F5-TTS pipeline-t használja, hogy a projekt `translated_splits` tartalmából készítsen új szinkronhangot. A referencia hangot normalizálja, opcionális EQ-t alkalmaz, majd Whisper-alapú verifikációval ellenőrzi a kimenetet.

## Kötelező beállítások
- `project_name` (pozícionális, alapértelmezés: nincs): A `workdir` alatti projekt könyvtár neve, amelyet fel kell dolgozni.
- `norm` (`--norm`, option, alapértelmezés: nincs): Normalizálási profil azonosító (pl. `hun`, `eng`) a szöveg előfeldolgozásához.

## Opcionális beállítások
- `model_dir` (`--model_dir`, option, alapértelmezés: nincs): A TTS modell könyvtárának útvonala. Üresen hagyva a szkript megpróbál automatikusan modellt keresni.
- `speed` (`--speed`, option, alapértelmezés: `1.0`): A generált hang sebessége (0.3–2.0 tartomány).
- `nfe_step` (`--nfe_step`, option, alapértelmezés: `32`): ODE lépésszám; nagyobb érték jobb minőségért, kisebb gyorsabb futásért.
- `remove_silence` (`--remove_silence`, flag, alapértelmezés: `false`): A generált wav-ok végeiről levágja a csendet.
- `phonetic_ref` (`--phonetic-ref`, flag, alapértelmezés: `false`): A referencia szöveget fonetikus átírással használja, ha telepítve van a modul.
- `normalize_ref_audio` (`--normalize-ref-audio`, flag, alapértelmezés: `false`): A referencia hanganyagot a `ref_audio_peak` szintre skálázza.
- `eq_config` (`--eq-config`, option, alapértelmezés: nincs): EQ görbét tartalmazó JSON (pl. `scripts/TTS/EQ.json`), amelyet a referencia audióra alkalmaz.
- `ref_audio_peak` (`--ref-audio-peak`, option, alapértelmezés: `0.95`): A referencia normalizálás cél csúcsértéke (0–1).
- `max_workers` (`--max_workers`, option, alapértelmezés: automatikus): A párhuzamos munkafolyamatok száma. Üresen hagyva a gép erőforrásaihoz igazodik.
- `seed` (`--seed`, option, alapértelmezés: `-1`): Véletlenmag a reprodukálhatósághoz. `-1` esetén véletlenszerűen generálja.
- `max_retries` (`--max-retries`, option, alapértelmezés: `5`): Hányszor próbálkozzon újragenerálással, ha a verifikáció elhasal.
- `tolerance_factor` (`--tolerance-factor`, option, alapértelmezés: `1.0`): Levenshtein-alapú ellenőrzés megengedett eltérésének szorzója a szó számával arányosan.
- `min_tolerance` (`--min-tolerance`, option, alapértelmezés: `2`): A tolerancia alsó küszöbe, hogy rövid szövegeknél se essen túl alacsonyra.
- `whisper_model` (`--whisper-model`, option, alapértelmezés: `openai/whisper-large-v3`): A verifikációhoz használt Whisper modell azonosítója.
- `beam_size` (`--beam-size`, option, alapértelmezés: `5`): A Whisper dekódolás nyalábszélessége.
- `save_failures` (`--save-failures`, flag, alapértelmezés: `false`): A sikertelen generálási kísérleteket menti diagnosztikai célra.
- `keep_best_over_tolerance` (`--keep-best-over-tolerance`, flag, alapértelmezés: `false`): Akkor is megtartja a legjobb kimenetet, ha az kicsivel túllépi a toleranciát.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózást kapcsol be a `tools.debug_utils` modul segítségével.
