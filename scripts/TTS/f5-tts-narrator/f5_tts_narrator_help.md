# f5_tts_narrator – konfigurációs útmutató

**Futtatási környezet:** `f5-tts`  
**Belépési pont:** `f5_tts_narrator.py`

A szkript egy narrátor referenciára építve generálja újra a projekt összes szinkron szegmensét F5-TTS modellel. A feldolgozás közben a referencia hangot normalizálja, opcionális EQ görbét alkalmaz, majd Whisper alapú verifikációval biztosítja a pontosságot.

## Kötelező beállítások
- `project_name` (pozícionális, alapértelmezés: nincs): A `workdir` alatt található projekt neve, amelynek szegmenseit generálni kell.
- `norm` (`--norm`, option, alapértelmezés: nincs): A normalizálási profil azonosítója (pl. `hun`, `eng`), amelyet a szöveg előfeldolgozásához használ.
- `narrator` (`--narrator`, option, alapértelmezés: nincs): A narrátor referencia könyvtár elérési útja; azonos nevű hang- és szövegfájlokat vár (`.wav/.mp3` + `.txt`).

## Opcionális beállítások
- `model_dir` (`--model_dir`, option, alapértelmezés: nincs): Az F5-TTS modell könyvtárának útvonala. Ha nincs megadva, a szkript interaktívan kérdez rá.
- `speed` (`--speed`, option, alapértelmezés: `1.0`): A generált hang időskálája (0.3–2.0 tartomány). Alacsonyabb érték lassabb, magasabb gyorsabb beszédet eredményez.
- `nfe_step` (`--nfe_step`, option, alapértelmezés: `32`): Az ODE integrátor lépésszáma. Nagyobb érték jobb minőséget adhat, de lassabb.
- `remove_silence` (`--remove_silence`, flag, alapértelmezés: `false`): Aktiválva a generált wav fájlokról automatikusan levágja a végeken lévő csendet.
- `phonetic_ref` (`--phonetic-ref`, flag, alapértelmezés: `false`): A referencia szöveget fonetikus átírásra alakítja, ha elérhető a `fonetikus_atiras` modul.
- `normalize_ref_audio` (`--normalize-ref-audio`, flag, alapértelmezés: `false`): A referencia hangot a `ref_audio_peak` értékre normalizálja, hogy konzisztens legyen a generálás.
- `eq_config` (`--eq-config`, option, alapértelmezés: nincs): EQ beállításokat tartalmazó JSON fájl (pl. `scripts/TTS/EQ.json`), amelyet a referencia audióra alkalmaz a generálás előtt.
- `ref_audio_peak` (`--ref-audio-peak`, option, alapértelmezés: `0.95`): A referencia audió normalizálásakor használt csúcsérték (0–1 között).
- `max_workers` (`--max_workers`, option, alapértelmezés: automatikus): A párhuzamosan futó munkafolyamatok száma. Üresen hagyva a szkript a CPU/GPU erőforrások alapján dönt.
- `seed` (`--seed`, option, alapértelmezés: `-1`): Véletlenmag a reprodukálhatósághoz. `-1` esetén minden szegmenshez új, random magot választ.
- `max_retries` (`--max-retries`, option, alapértelmezés: `5`): Hányszor próbálja újragenerálni a szegmenst, ha a verifikációs lépés nem sikerül.
- `tolerance_factor` (`--tolerance-factor`, option, alapértelmezés: `1.0`): Levenshtein-alapú ellenőrzéshez használt szorzó, amely megadja, hány szónyi eltérés megengedett a generált szövegben.
- `min_tolerance` (`--min-tolerance`, option, alapértelmezés: `2`): A dinamikusan számított tolerancia alsó korlátja, így rövid szövegeknél sem esik nullára.
- `whisper_model` (`--whisper-model`, option, alapértelmezés: `openai/whisper-large-v3`): A verifikációhoz használt Whisper modell neve (HF azonosító).
- `beam_size` (`--beam-size`, option, alapértelmezés: `5`): A Whisper dekódolás nyalábszélessége a verifikációs lépésben.
- `save_failures` (`--save-failures`, flag, alapértelmezés: `false`): Elmenti a sikertelen generálási kísérletek hangját és metaadatait egy `failed_generations` mappába diagnosztikához.
- `keep_best_over_tolerance` (`--keep-best-over-tolerance`, flag, alapértelmezés: `false`): Ha minden próbálkozás a tolerancia felett marad, akkor is elmenti a legjobb (legkisebb távolságú) változatot.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózást kapcsol be a `tools.debug_utils` modulon keresztül.
