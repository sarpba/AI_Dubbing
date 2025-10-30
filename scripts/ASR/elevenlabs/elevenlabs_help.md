# evenlabs – konfigurációs útmutató

**Futtatási környezet:** `sync`
**Belépési pont:** `elevenlabs.py`

A szkript az Evenlabs speech-to-text API-t hívja meg, és a projekt `separated_audio_speech` mappájában található hangfájlokhoz normalizált `word_segments` listát ment el. Minden szó objektum tartalmazza a szót, a kezdő- és végidőt (másodpercben), opcionális pontossági értéket és – ha elérhető – a beszélő azonosítót.

> **Fontos:** A keletkező `word_segments` JSON-t a következő lépésben a `scripts/ASR/resegment/resegment.py` szkripttel szükséges mondatszintre újraszegmentálni, ezért ezt a lépést illeszd be közvetlenül az elevenlabs futtatása után.

Példa kimenet:

```json
{
  "word_segments": [
    {
      "word": "If",
      "start": 1.28,
      "end": 1.46,
      "score": null,
      "speaker": "speaker_0"
    }
  ],
  "language": "en",
  "provider": "evenlabs",
  "diarization": true
}
```

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatti projektmappa neve, amelynek hanganyagát feldolgozza.

## Opcionális beállítások
- `language` (`--language`, option, alapértelmezés: nincs): ISO nyelvkód (pl. `en`, `hu`). Ha nincs megadva, az Evenlabs automatikus nyelvfelismerésével dolgozik.
- `model` (`--model`, option, alapértelmezés: `scribe_v1_experimental`): Az Evenlabs ASR modell azonosítója.
- `api_url` (`--api-url`, option, alapértelmezés: `https://api.elevenlabs.io/v1/speech-to-text`): A használni kívánt Evenlabs STT végpont.
- `api_key` (`--api-key`, option, alapértelmezés: nincs): Evenlabs API kulcs. Megadáskor elmenti base64 formában a `keyholder.json` fájlba. Alternatíva az `EVENLABS_API_KEY` környezeti változó.
- `diarize` (`--diarize` / `--no-diarize`, flag, alapértelmezés: `--diarize`): A diarizáció be- vagy kikapcsolása az API kérésben.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás, figyelmeztetések megjelenítése.

> **Megjegyzés:** A szkript minden feldolgozott hangfájllal azonos nevű `.json` fájlt hoz létre ugyanabban a mappában, amely az ElevenLabs válaszából származó normalizált `word_segments` listát tartalmazza.
