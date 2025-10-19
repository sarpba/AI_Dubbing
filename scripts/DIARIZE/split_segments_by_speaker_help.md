# split_segments_by_speaker_codex – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**API:** `huggingface`  
**Belépési pont:** `DIARIZE/split_segments_by_speaker.py`

A szkript a pyannote `speaker-diarization-3.1` pipeline-ját használja a már meglévő időzített szegmensek beszélőkhöz rendelésére. A meglévő JSON szegmenseket kibővíti vagy szétválasztja a diarizáció eredménye alapján, opcionális biztonsági mentéssel és naplózással.

## Kötelező beállítás
- `project` (`-p`, `--project`, option, alapértelmezés: nincs): A `workdir` alatti projektmappa neve, amelynek beszédszegmenseit módosítani kell.

## Opcionális beállítások
- `hf_token` (`--hf-token`, option, alapértelmezés: nincs): Hugging Face token a zárt pyannote modellek eléréséhez. Token nélkül a pipeline nem futtatható.
- `audio_exts` (`--audio-exts`, option, alapértelmezés: `wav,flac,mp3,m4a`): Vesszővel elválasztott lista, amely megadja, milyen kiterjesztésű audiókra keressen diarizációs alapanyagot.
- `min_chunk` (`--min-chunk`, option, alapértelmezés: `0.2`): A minimális szegmenshossz másodpercben. Ennél rövidebb futásokat nem próbál szétválasztani.
- `round` (`--round`, option, alapértelmezés: `2`): A kimeneti időbélyegek kerekítése ennyi tizedesjegyre.
- `add_speaker_field` (`--no-speaker-field`, flag, alapértelmezés: `true`): Alapból minden szegmens kap `speaker` mezőt. A `--no-speaker-field` kapcsolóval ezt elkerülheted.
- `backup` (`--no-backup`, flag, alapértelmezés: `true`): Ha engedélyezett és a feldolgozás felülírja a bemeneti JSON-t, `.bak` biztonsági másolatot készít. A `--no-backup` kapcsolóval tiltható.
- `dry_run` (`--dry-run`, flag, alapértelmezés: `false`): Próba futás; csak naplóban jelzi, mit módosítana, tényleges fájlírás nélkül.
- `min_word_overlap` (`--min-word-overlap`, option, alapértelmezés: `0.5`): Szóalapú hozzárendelésnél a minimális átfedési arány, ha a középpont alapú módszer nem egyértelmű.
- `inplace` (`--no-inplace`, flag, alapértelmezés: `true`): Alapállapotban a bemeneti JSON-t módosítja. A `--no-inplace` kapcsolóval új fájlba (`<név><suffix>.json`) ír.
- `output_suffix` (`--output-suffix`, option, alapértelmezés: `_split`): A kimeneti fájl utótagja, ha nem helyben írja felül a bemenetet.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletesebb naplózást aktivál.
