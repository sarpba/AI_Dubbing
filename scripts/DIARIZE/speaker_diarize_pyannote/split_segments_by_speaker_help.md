# split_segments_by_speaker

**Futtatási környezet:** `sync`  
**Belépési pont:** `DIARIZE/speaker_diarize_pyannote/split_segments_by_speaker.py`

## Mit csinál?
A beszéd szegmenseket beszélőnként szétválasztja.

A script a beszélőkhöz kapcsolódó információkat állít elő vagy finomítja, hogy a szegmensek beszélő szerint is kezelhetők legyenek.

## Kötelező paraméterek
- `project` (opció;  kapcsoló: `-p`, `--project`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `hf_token` (opció;  kapcsoló: `--hf-token`; alapértelmezés: nincs): Hugging Face token a védett modellek vagy diarizációs komponensek eléréséhez.
- `audio_exts` (opció;  kapcsoló: `--audio-exts`; alapértelmezés: `wav,flac,mp3,m4a`): A feldolgozható hangfájl-kiterjesztések listája.
- `min_chunk` (opció;  kapcsoló: `--min-chunk`; alapértelmezés: `0.2`): A létrejövő beszélői szegmensek minimális hossza.
- `round` (opció;  kapcsoló: `--round`; alapértelmezés: `2`): A kerekítés pontossága az időbélyegeknél.
- `add_speaker_field` (kapcsoló;  kapcsoló: `--no-speaker-field`; alapértelmezés: `true`): Alapból bekapcsolt. A kimeneti JSON-ba beleírja a beszélő azonosítót is. Alapállapotban be van kapcsolva; a negatív kapcsolóval kikapcsolható.
- `backup` (kapcsoló;  kapcsoló: `--no-backup`; alapértelmezés: `true`): A script biztonsági mentést készít a módosított fájlokról a feldolgozás előtt. Alapállapotban be van kapcsolva; a negatív kapcsolóval kikapcsolható.
- `dry_run` (kapcsoló;  kapcsoló: `--dry-run`; alapértelmezés: `false`): Próbafutást végez fájlmódosítás nélkül. Alapállapotban ki van kapcsolva.
- `min_word_overlap` (opció;  kapcsoló: `--min-word-overlap`; alapértelmezés: `0.5`): A szavak és a beszélői szakaszok egyeztetésének küszöbértéke.
- `inplace` (kapcsoló;  kapcsoló: `--no-inplace`; alapértelmezés: `true`): Alapból bekapcsolt. A script az eredeti fájlt módosítja ahelyett, hogy külön másolatot készítene. Alapállapotban be van kapcsolva; a negatív kapcsolóval kikapcsolható.
- `output_suffix` (opció;  kapcsoló: `--output-suffix`; alapértelmezés: `_split`): A külön fájlba írt kimenetek utótagja.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
