# whisx2 – konfigurációs útmutató

**Futtatási környezet:** `whisperx`  
**API:** `huggingface`  
**Belépési pont:** `ASR/whisx_v1.1.py`

A szkript a WhisperX `large-v3` modellt futtatja több GPU-n, opcionális beszélő‑diarizációval. A Parakeet szkriptből átemelt, finomhangolható szegmentálási logikát használ, hogy a `separated_audio_speech` mappában található hangokról időbélyeges JSON átiratokat készítsen. A kimeneteket az audiók mellé menti.

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `config.json` alapján felépített `workdir/<projekt>` mappa neve. Ez határozza meg, hogy melyik projekt hanganyagai kerülnek feldolgozásra.

## Opcionális beállítások
- `gpus` (`--gpus`, option, alapértelmezés: automatikus): Vesszővel tagolt CUDA eszköz indexek listája (például `0,1`). Ha nincs megadva, a szkript a `nvidia-smi` alapján az összes elérhető GPU-t használja.
- `hf_token` (`--hf_token`, option, alapértelmezés: nincs): Hugging Face hozzáférési token a pyannote diarizációs modellhez. Token nélkül az átirat elkészül, de beszélő szerinti szétosztás nem történik.
- `language` (`--language`, option, alapértelmezés: automatikus): ISO nyelvkód (pl. `en`, `hu`). Megadásával fix nyelvre kényszeríthető a modell; üresen hagyva automatikusan detektálja.
- `max_pause` (`--max-pause`, option, alapértelmezés: `0.6`): Maximális szünet a mondatok között másodpercben; nagyobb érték kevésbé darabol.
- `timestamp_padding` (`--timestamp-padding`, option, alapértelmezés: `0.2`): Időbélyegek kiterjesztése másodpercben, hogy ne vágja le a beszéd elejét/végét.
- `max_segment_duration` (`--max-segment-duration`, option, alapértelmezés: `11.5`): Egy szegmens felső hosszkorlátja másodpercben (`0` = nincs limit).

## Működés és kimenet
- Bemenet/kimenet könyvtár: `workdir/<projekt>/separated_audio_speech` a `config.json` alapján.
- Minden feldolgozott hangfájl mellé egy azonos nevű `.json` készül a felismert `language`, a mondat szegmensek (`segments`) és a szó szintű időbélyegek (`word_segments`) mezőkkel.
- Már létező JSON mellett a fájl kihagyásra kerül.
