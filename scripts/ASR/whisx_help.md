# whisx – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**API:** `huggingface`  
**Belépési pont:** `ASR/whisx.py`

A szkript a WhisperX modellt futtatja GPU-kon, hogy a `separated_audio_speech` mappában található hangfájlokból átírt és diarizált JSON kimenetet készítsen. Több GPU esetén minden eszközre külön folyamatot indít, a kimeneteket az audió mellé menti.

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `config.json` alapján felépített `workdir/<projekt>` mappa neve. Ez határozza meg, hogy melyik projekt hanganyagai kerülnek feldolgozásra.

## Opcionális beállítások
- `gpus` (`--gpus`, option, alapértelmezés: automatikus): Vesszővel tagolt CUDA eszköz indexek listája (például `0,2,3`). Ha üresen hagyod, a szkript a `nvidia-smi` kimenete alapján az összes elérhető GPU-t kihasználja.
- `hf_token` (`--hf_token`, option, alapértelmezés: nincs): Hugging Face hozzáférési token a pyannote diarizációs modellhez. Token nélkül az átirat elkészül, de beszélő szerinti szétosztás nem történik.
- `language` (`--language`, option, alapértelmezés: automatikus): ISO nyelvkód (pl. `en`, `de`, `hu`). Megadásával fix nyelvre kényszerítheted a modellt; üresen hagyva a WhisperX automatikusan detektálja a nyelvet.
