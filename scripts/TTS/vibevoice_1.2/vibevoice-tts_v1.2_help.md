# vibevoice-tts_v1.2 – konfigurációs útmutató

**Futtatási környezet:** `vibevoice`  
**Belépési pont:** `vibevoice-tts_v1.2.py`

 A szkript a VibeVoice TTS modellt és opcionálisan LoRA adaptereket használ, hogy a projekt `translated_splits` kimenetét új magyar hangra szintetizálja. A referencia sávot szegmensekre bontja, EQ-t és normalizálást alkalmazhat, Whisper-alapú visszaellenőrzést végez Levenshtein-távolság alapján, és több GPU-t is képes párhuzamosan kihasználni. A `v1.2` frissítés opcionális pitch összehasonlítást épít be: a generált szegmensek hangmagasságát a referenciához viszonyítja, szükség esetén újragenerál, és külön retry keretet biztosít a pitch hibák kezelésére, miközben megőrzi a `v1.1` több-GPU és diagnosztikai fejlesztéseit.

## Kötelező beállítások
- `project_name` (pozícionális, alapértelmezés: nincs): A `workdir` alatti projekt könyvtár neve, amelynek szegmenseit feldolgozzuk.
- `norm` (`--norm`, option, alapértelmezés: nincs): Normalizálási profil azonosító (pl. `hun`, `eng`). A szkript ehhez igazítja a szöveg-előkészítést és a Whisper nyelvi beállítását.

## Opcionális beállítások
- `model_path` (`--model_path`, option, alapértelmezés: `microsoft/VibeVoice-1.5b`): Hugging Face modellazonosító vagy lokális mappa az alap VibeVoice modellhez.
- `model_dir` (`--model_dir`, option, alapértelmezés: nincs): Lokális könyvtár megadása; elsőbbséget élvez a `--model_path` értékével szemben.
- `checkpoint_path` (`--checkpoint_path`, option, alapértelmezés: nincs): LoRA / adapter útvonal, amelyet a modellre töltünk.
- `device` (`--device`, option, alapértelmezés: automatikus): Cél eszköz (`cuda`, `mps`, `cpu`). Több GPU esetén a szkript automatikusan képes párhuzamosítani.
- `cfg_scale` (`--cfg_scale`, option, alapértelmezés: `1.3`): Classifier-Free Guidance skála; nagyobb érték erősebb stílus-követést eredményez.
- `disable_prefill` (`--disable_prefill`, flag, alapértelmezés: `false`): Kikapcsolja a voice cloning jellegű prefill lépést (`is_prefill=False`).
- `ddpm_steps` (`--ddpm_steps`, option, alapértelmezés: `10`): A diffúziós inference lépések száma; emelése jobb minőséget, hosszabb futást ad.
- `eq_config` (`--eq_config`, option, alapértelmezés: `scripts/TTS/EQ.json` ha létezik): EQ görbét tartalmazó JSON, amellyel a referencia audiót korrigáljuk.
- `normalize_ref_audio` (`--normalize_ref_audio`, flag, alapértelmezés: `false`): A referencia mintát a `ref_audio_peak` szintre normálja.
- `ref_audio_peak` (`--ref_audio_peak`, option, alapértelmezés: `0.95`): A normalizálási cél csúcsértéke 0–1 tartományban.
- `target_sample_rate` (`--target_sample_rate`, option, alapértelmezés: `16000`): A referencia audiót erre a mintavételi frekvenciára reszámplálja a TTS előtt; 0 vagy negatív értékkel kikapcsolható.
- `speaker_name` (`--speaker_name`, option, alapértelmezés: `Speaker 1`): A VibeVoice által elvárt „Speaker X:” formátumhoz használt címke. A szöveg elejére automatikusan rákerül, ha hiányzik.
- `max_retries` (`--max_retries`, option, alapértelmezés: `5`): Whisper alapú ellenőrzés esetén ennyiszer próbálkozik újragenerálással.
- `enable_pitch_check` (`--enable_pitch_check`, flag, alapértelmezés: `false`): Bekapcsolja a pitch összehasonlítást; a referencia és a generált szegmens medián hangmagasságát veti össze, és tolerancián kívül újra próbálkozik.
- `pitch_tolerance` (`--pitch_tolerance`, option, alapértelmezés: `20.0`): Megengedett medián hangmagasság eltérés (Hz) a referencia és a generált szegmens között.
- `pitch_min_frequency` (`--pitch_min_frequency`, option, alapértelmezés: `60.0`) és `pitch_max_frequency` (`--pitch_max_frequency`, option, alapértelmezés: `400.0`): A pitch detektálás keresési tartománya (Hz), a narrátor várható beszédmagasságához igazítva.
- `pitch_retry` (`--pitch_retry`, option, alapértelmezés: `3`): Pitch hibák esetén ennyi extra próbálkozást enged (3 → összesen 4 kísérlet a szegmensre).
- `tolerance_factor` (`--tolerance_factor`, option, alapértelmezés: `1.0`) és `min_tolerance` (`--min_tolerance`, option, alapértelmezés: `2`): A Levenshtein-távolság megengedett mértékét szabályozzák (szó-mennyiség * faktor, minimum érték).
- `whisper_model` (`--whisper_model`, option, alapértelmezés: `openai/whisper-large-v3`) és `beam_size` (`--beam_size`, option, alapértelmezés: `5`): A visszaellenőrzéshez használt ASR modell és beamszélesség.
- `seed` (`--seed`, option, alapértelmezés: `-1`): Véletlenmag. `-1` esetén véletlenszerűen indul, Whisperrel visszaellenőriz, és szükség esetén újra próbálkozik. Pozitív értéknél determinisztikus (visszaellenőrzés nélkül).
- `save_failures` (`--save_failures`, flag, alapértelmezés: `false`): Sikertelen kimenetek és metaadatok mentése a `failed_generations` mappába; pitch ellenőrzésnél a mentett wav fájl neve tartalmazza az eltérés nagyságát.
- `keep_best_over_tolerance` (`--keep_best_over_tolerance`, flag, alapértelmezés: `false`): Ha nincs tolerancián belüli találat, a legjobb (legkisebb távolságú) próbálkozást akkor is megtartja.
- `max_segments` (`--max_segments`, option, alapértelmezés: nincs): Debug célra limitálja a feldolgozott szegmensek számát.
- `max_workers` (`--max_workers`, option, alapértelmezés: automatikus): Több GPU esetén limitálja a párhuzamos workerek számát.
- `overwrite` (`--overwrite`, flag, alapértelmezés: `false`): Létező kimeneti wav fájlok felülírása.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás a `tools.debug_utils` modul segítségével.

## Bemeneti könyvtár felülbírálása
- `input_directory_override` (`--input_directory_override`, option, alapértelmezés: `HAGYD_URESEN_ALAPERTELMEZESKENT`): Opcionálisan megadható könyvtár, ahonnan az első `.wav` és `.json` fájlokat keresi.
- Az alapértelmezett sztring azt jelzi, hogy hagyd üresen/érintetlenül, így a projekt `workdir` struktúrájából dolgozik továbbra is.
- Abszolút és projektgyökérhez viszonyított útvonal is megadható; utóbbi automatikusan a teljes elérési útra egészül ki.
- Nem létező könyvtár esetén figyelmeztetést naplóz, majd visszavált az alapértelmezett mappákra.
