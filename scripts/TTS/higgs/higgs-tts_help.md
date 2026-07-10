# higgs-tts

**Futtatási környezet:** `higgs-tts-gradio`  
**Belépési pont:** `TTS/higgs/higgs-tts.py`

## Mit csinál?
A Higgs TTS 3 modellt használva hozza létre a szinkron szegmenseket a fordított állományból.

A script a `vibevoice_1.2` projektintegrációs mintáját követi: a projekt `translated` JSON-jából dolgozik, a `2_separated_audio_speech` forrás wav-ból vágja ki a szegmensenkénti referenciahangot, majd a kész hangokat a `translated_splits` mappába menti.

Kiemelt működés:
- a `bosonai/higgs-tts-3-4b` alias lokálisan automatikusan a Transformers-kompatibilis `multimodalart/higgs-audio-v3-tts-4b-transformers` modellre oldódik;
- a `2.0` másodpercnél rövidebb referenciahangokat alapértelmezés szerint ismétléssel legalább `4.0` másodpercre bővíti a stabilabb voice cloning érdekében;
- a `vibevoice_1.2` scriptben meglévő Whisper/pitch alapú hibajavító és újrapróbálkozó mechanizmusokat nem tartalmazza.

## Kötelező paraméterek
- `project_name` (pozicionális; kapcsoló: pozicionális; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.
- `norm` (opció; kapcsoló: `--norm`; alapértelmezés: nincs): A használt normalizálási profil neve.

## Opcionális paraméterek
- `model_path` (opció; kapcsoló: `--model_path`; alapértelmezés: `bosonai/higgs-tts-3-4b`): A betöltendő modell aliasa vagy Hugging Face azonosítója.
- `model_dir` (opció; kapcsoló: `--model_dir`; alapértelmezés: nincs): Lokális modell snapshot vagy könyvtár. Ha meg van adva, ezt használja a script.
- `device` (opció; kapcsoló: `--device`; alapértelmezés: `auto`): A futtatás eszköze, például `cpu`, `cuda`, `cuda:0` vagy `mps`.
- `temperature` (opció; kapcsoló: `--temperature`; alapértelmezés: `0.8`): Mintavételi hőmérséklet.
- `top_p` (opció; kapcsoló: `--top_p`; alapértelmezés: `0.95`): Top-p mintavétel.
- `top_k` (opció; kapcsoló: `--top_k`; alapértelmezés: `50`): Top-k mintavétel.
- `max_new_tokens` (opció; kapcsoló: `--max_new_tokens`; alapértelmezés: `1024`): Maximális generált tokenek száma.
- `seed` (opció; kapcsoló: `--seed`; alapértelmezés: `-1`): Fix véletlenmag. `-1` esetén nincs rögzítve.
- `eq_config` (opció; kapcsoló: `--eq_config`; alapértelmezés: nincs): EQ beállításokat tartalmazó JSON fájl a referenciahang előkészítéséhez.
- `normalize_ref_audio` (kapcsoló; kapcsoló: `--normalize_ref_audio`; alapértelmezés: `false`): A referenciahangot egységes peak szintre normalizálja. Alapállapotban ki van kapcsolva.
- `ref_audio_peak` (opció; kapcsoló: `--ref_audio_peak`; alapértelmezés: `0.95`): A referenciahang cél peak értéke normalizáláskor.
- `disable_short_ref_expansion` (kapcsoló; kapcsoló: `--disable_short_ref_expansion`; alapértelmezés: `false`): Kikapcsolja a rövid referenciahangok automatikus ismétlését. Alapállapotban ki van kapcsolva.
- `short_ref_threshold` (opció; kapcsoló: `--short_ref_threshold`; alapértelmezés: `2.0`): Ez alatti referenciahangokat ismétli a script.
- `short_ref_target_seconds` (opció; kapcsoló: `--short_ref_target_seconds`; alapértelmezés: `4.0`): A rövid referenciahang bővítésének célhossza másodpercben.
- `emotion` (opció; kapcsoló: `--emotion`; alapértelmezés: `none`): Globális `emotion` token, amely minden generált prompt elé bekerül.
- `style` (opció; kapcsoló: `--style`; alapértelmezés: üres): Globális `style` token érték.
- `speed` (opció; kapcsoló: `--speed`; alapértelmezés: `none`): Globális `prosody` speed token.
- `pitch` (opció; kapcsoló: `--pitch`; alapértelmezés: `none`): Globális `prosody` pitch token.
- `expressive` (opció; kapcsoló: `--expressive`; alapértelmezés: `none`): Globális `prosody` expressive token.
- `manual_prefix` (opció; kapcsoló: `--manual_prefix`; alapértelmezés: üres): Kézzel megadott prompt prefix tokenek, például `<|prosody:pause|>`.
- `raw_json` (opció; kapcsoló: `--raw_json`; alapértelmezés: üres): Opcionális JSON felülírás a `generation` paraméterekhez (`temperature`, `top_p`, `top_k`, `max_new_tokens`).
- `input_directory_override` (kapcsoló; kapcsoló: `--input_directory_override`; alapértelmezés: `false`): A fordított JSON-okat a `temp` mappából olvassa. Alapállapotban ki van kapcsolva.
- `max_segments` (opció; kapcsoló: `--max_segments`; alapértelmezés: nincs): A feldolgozható szegmensek számát limitálja.
- `overwrite` (kapcsoló; kapcsoló: `--overwrite`; alapértelmezés: `false`): A már meglévő kimeneti wav fájlokat is újragenerálja. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló; kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be. Alapállapotban ki van kapcsolva.

## Példa futtatás

```bash
python scripts/TTS/higgs/higgs-tts.py demo_project \
  --norm hun \
  --device cuda:0 \
  --normalize_ref_audio \
  --temperature 0.8 \
  --top_p 0.95
```

## Megjegyzés
A script a referencia transcripthez elsősorban a szegmens `text` mezőjét használja. Ha ez hiányzik, a voice cloning így is lefut, csak transcript nélküli referenciával.
