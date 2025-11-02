# canary-easy – konfigurációs útmutató

**Futtatási környezet:** `nemo`  
**Belépési pont:** `canary-easy.py`

A szkript az NVIDIA Canary ASR modellt futtatja a `separated_audio_speech` mappában található hangfájlokon. A felvételeket darabolja, majd a Parakeet szkriptjéhez igazodó, szóalapú időbélyegekkel ellátott JSON-t ment (`segments` + `word_segments` mezők).

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatt található projektmappa neve, amelynek hanganyagát fel kell dolgozni.

## Opcionális beállítások
- `source_lang` (`--source-lang`, option, alapértelmezés: `auto`): Forrásnyelv kódja; `auto` esetén a Canary próbálja felismerni. Kétbetűs nyelvi kódot kell használni, pl. `hu`
- `target_lang` (`--target-lang`, option, alapértelmezés: nincs): Célnyelv kódja, ha a Canary fordítson is. Üresen hagyva csak átírás készül. Kétbetös nyelvi kódot kell használni, pl. `hu` ha a forrással azonos nyelvet adunk meg, akkor beolvas aa modell, ha mást akkor fordít. A fordítás csak angol nyelvre van tréningezve. Fordítás feállítása esetén angolon kívül mást nem javasolt használni.
- `model_name` (`--model-name`, option, alapértelmezés: `nvidia/canary-1b-v2`): A használandó Canary modell Hugging Face azonosítója.
- `batch_size` (`--batch-size`, option, alapértelmezés: `4`): Hány chunk kerüljön egyszerre GPU-ra. (Jelenleg a szóalapú időbélyegek stabilitása érdekében a szkript egydarabos batch-sel fut, így ez a kapcsoló inkább jövőbeni kompatibilitás miatt maradt meg.)
- `beam_size` (`--beam-size`, option, alapértelmezés: `5`): Beam-search szélessége a dekóderben.
- `len_pen` (`--len-pen`, option, alapértelmezés: `1.0`): Hossz-büntetés a dekóderben; 1 felett a rövidebb, 1 alatt a hosszabb hipotéziseket részesíti előnyben.
- `chunk` (`--chunk`, option, alapértelmezés: `30`): A chunk maximális hossza másodpercben (10–120 mp tartomány javasolt); a szkript a határ előtt legfeljebb 1 mp-nyi visszatekintésben keres legalább 0,2 mp tétlencsendet, és ha talál, ott vágja el a chunkot, csökkentve a szóvégek darabolását.
- `max_pause` (`--max-pause`, option, alapértelmezés: `0.6`): A szóközi szünet, amely fölött új mondatszegmens indul a kimenetben.
- `timestamp_padding` (`--timestamp-padding`, option, alapértelmezés: `0.2`): Ennyivel tolja ki a szegmentált szavak elejét/végét, hogy jobban fedjék a beszédet.
- `max_segment_duration` (`--max-segment-duration`, option, alapértelmezés: `11.5`): A `segments` bejegyzések maximális hossza másodpercben (0 = nincs limit).

- `keep_alternatives` (`--keep-alternatives`, option, alapértelmezés: `2`): A dekóder alternatív hipotéziseinek száma chunkonként; a Parakeet-kompatibilis kimenet jelenleg ezeket nem menti el.
- `overwrite` (`--overwrite`, flag, alapértelmezés: `false`): Ha engedélyezed, felülírja a már létező kimeneti fájlokat; ellenkező esetben kihagyja őket.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás a `tools.debug_utils` segítségével.
