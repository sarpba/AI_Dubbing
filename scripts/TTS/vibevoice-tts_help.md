# vibevoice-tts – konfigurációs útmutató

**Futtatási környezet:** `vibevoice`  
**Belépési pont:** `TTS/vibevoice-tts.py`

A szkript a VibeVoice TTS modellt és opcionálisan LoRA adaptereket használ, hogy a projekt `translated_splits` kimenetét új magyar hangra szintetizálja. A referencia sávot szegmensekre bontja, EQ-t és normalizálást alkalmazhat, majd a VibeVoice generátoron keresztül wav fájlokat készít.

## Kötelező beállítások
- `project_name` (pozícionális, alapértelmezés: nincs): A `workdir` alatti projekt könyvtár neve, amelynek szegmenseit feldolgozzuk.

## Opcionális beállítások
- `model_path` (`--model_path`, option, alapértelmezés: `microsoft/VibeVoice-1.5b`): Hugging Face modellazonosító vagy lokális mappa az alap VibeVoice modellhez.
- `model_dir` (`--model_dir`, option, alapértelmezés: nincs): Alternatív megadás lokális modell-könyvtárra; elsőbbséget élvez a `--model_path` értékével szemben.
- `checkpoint_path` (`--checkpoint_path`, option, alapértelmezés: nincs): LoRA vagy finomhangolt adapter könyvtár útvonala, amelyet a modellre töltünk.
- `device` (`--device`, option, alapértelmezés: automatikus): Cél eszköz (`cuda`, `mps`, `cpu`). Üresen hagyva a szkript elérhetőség alapján választ.
- `cfg_scale` (`--cfg_scale`, option, alapértelmezés: `1.3`): Classifier-Free Guidance skála; nagyobb érték erősebb stílus-követést eredményez.
- `disable_prefill` (`--disable_prefill`, flag, alapértelmezés: `false`): Kikapcsolja a voice cloning jellegű prefill lépést (`is_prefill=False`).
- `ddpm_steps` (`--ddpm_steps`, option, alapértelmezés: `10`): A diffúziós inference lépések száma; emelése jobb minőséget, hosszabb futást ad.
- `eq_config` (`--eq_config`, option, alapértelmezés: `scripts/TTS/EQ.json` ha létezik): EQ görbét tartalmazó JSON, amellyel a referencia audiót korrigáljuk.
- `normalize_ref_audio` (`--normalize_ref_audio`, flag, alapértelmezés: `false`): A referencia mintát a `ref_audio_peak` szintre normálja.
- `ref_audio_peak` (`--ref_audio_peak`, option, alapértelmezés: `0.95`): A normalizálási cél csúcsértéke 0–1 tartományban.
- `speaker_name` (`--speaker_name`, option, alapértelmezés: `Speaker 1`): A VibeVoice által elvárt „Speaker X:” formátumhoz használt címke. A megadott értékkel egészíti ki azokat a szegmenseket, ahol nincs beszélő megadva.
- `seed` (`--seed`, option, alapértelmezés: `0`): Véletlenmag. `0` esetén nem fixálja, így minden futás más eredményt adhat.
- `max_segments` (`--max_segments`, option, alapértelmezés: nincs): Debug célra limitálja a feldolgozott szegmensek számát.
- `overwrite` (`--overwrite`, flag, alapértelmezés: `false`): Létező kimeneti wav fájlok felülírása.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás a `tools.debug_utils` modul segítségével.
