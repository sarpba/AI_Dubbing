# vibevoice-tts_narrátor – konfigurációs útmutató

**Futtatási környezet:** `vibevoice`  
**Belépési pont:** `TTS/vibevoice-tts_narrator.py`

A szkript a standard `vibevoice-tts` folyamatot bővíti azzal, hogy a narrátor hangmintát nem a fordított sávokból vágja ki, hanem egy külön megadott könyvtárban található, egyetlen WAV fájlból tölti be. A betöltött referencia hangra ugyanazokat az EQ, normalizálási és újramintavételezési lépéseket alkalmazza, majd minden szegmenshez ugyanazt a mintát használja a VibeVoice TTS modell betanításához.

## Kötelező beállítások
- `project_name` (pozícionális, alapértelmezés: nincs): A `workdir` alatti projekt azonosítója, amelynek szegmenseit feldolgozzuk.
- `norm` (`--norm`, option, alapértelmezés: nincs): Normalizálási profil (pl. `hun`, `eng`). Meghatározza a szöveg-előkészítést és a Whisper ellenőrzés nyelvét.
- `narrator` (`--narrator`, option, alapértelmezés: nincs): A narrátor referencia könyvtár elérési útja. A mappában pontosan egy darab `.wav` fájlnak kell lennie; ezt a szkript ideiglenes fájlba konvertálja, és minden szegmenshez ezt használja hangmintaként.

## Opcionális beállítások
- `model_path` / `model_dir`: A VibeVoice modell vagy lokális könyvtár elérési útja. Ha mindkettő meg van adva, a `model_dir` élvez elsőbbséget.
- `checkpoint_path`: LoRA vagy adapter könyvtár, amelyet betölt a modellre.
- `device`: Cél eszköz (`cuda`, `mps`, `cpu`, `auto`). Több GPU esetén automatikusan szétosztja a szegmenseket.
- `cfg_scale`, `disable_prefill`, `ddpm_steps`: A VibeVoice generálás finomhangolása (CF guidance, prefill kikapcsolása, diffúziós lépések száma).
- `eq_config`: EQ görbét tartalmazó JSON. Ha létezik, a narrátor mintára alkalmazzuk, mielőtt ideiglenes fájlba írnánk.
- `normalize_ref_audio`, `ref_audio_peak`: A narrátor referencia hang normalizálása a megadott csúcsértékre.
- `target_sample_rate`: Ha >0, a narrátor mintát erre a mintavételi frekvenciára reszámpláljuk.
- `speaker_name`: A beszélő prefixe, amelyet a szkript automatikusan beilleszt a generálandó szöveg elejére, ha hiányzik.
- `seed`, `max_retries`, `tolerance_factor`, `min_tolerance`, `whisper_model`, `beam_size`: A Whisper-alapú visszaellenőrzés és újragenerálás paraméterei, megegyeznek az alap szkripttel.
- `max_segments`, `overwrite`, `save_failures`, `keep_best_over_tolerance`, `max_workers`, `debug`: Debug és futtatási beállítások a feldolgozás limitálására, logolásra, illetve hibás kimenetek mentésére.

## Folyamat áttekintése
1. A projektből betölti a `translated` JSON-t, és előkészíti a kimeneti mappákat (`translated_splits`, `noice_splits`).
2. Kiválasztja a narrátor könyvtár egyetlen WAV fájlját, alkalmazza az EQ/normalizálás/reszámplálás lépéseket, majd ideiglenes fájlba menti.
3. (Opcionálisan) a projekt eredeti hivatkozó sávjából kimenti a zaj-szegmenseket változatlan módon.
4. A szegmenseket a kiválasztott eszköz(ök)re osztja ki, minden szegmensnél ugyanazzal a narrátor mintával futtatja a VibeVoice modellt, majd Whisperrel ellenőrzi az eredményt (ha `--seed -1`).
5. A sikeres szegmenseket beírja a `translated_splits` mappába, opcionálisan menti a sikertelen próbálkozásokat diagnosztikai célra.

## Hasznos tippek
- Ügyelj rá, hogy a narrátor könyvtárban tényleg csak egy `.wav` fájl legyen, különben a szkript hibát dob.
- Hosszabb narrátor mintát is használhatsz; a szkript csak egyszer tölti be és előkészíti, majd minden worker ugyanazt az ideiglenes fájlt olvassa.
- Ha új EQ vagy normalizálási beállításokat próbálnál ki, egyszerűen add meg a megfelelő flaget; a narrátor referencia újra lesz generálva a futtatás elején.
- Több GPU esetén érdemes `--max_workers` értéket megadni, hogy kontrolláld az egyidejű példányok számát.
