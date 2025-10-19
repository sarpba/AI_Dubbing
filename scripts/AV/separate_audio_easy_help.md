# separate_audio_easy_codex – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `AV/separate_audio_easy.py`

A szkript Demucs/MDX modellekkel szétválasztja a videóból kivont audiót beszédre és háttér-sávra. Kezeli a több modellből származó eredmények összefésülését, chunk-alapú feldolgozást és opcionális háttérkeverést.

## Kötelező beállítás
- `project` (`-p`, `--project`, option, alapértelmezés: nincs): A `workdir` alatti projekt neve, amelynek audióját szét kell választani.

## Opcionális beállítások
- `device` (`--device`, option, alapértelmezés: `cuda`): A futtatandó eszköz (`cuda`/`cpu`). Ha GPU nem érhető el, automatikusan CPU-ra vált.
- `models` (`--models`, option, alapértelmezés: `htdemucs,mdx_extra`): Vesszővel elválasztott modellnevek. A szkript ebben a sorrendben próbálja ki őket, majd kombinálja az eredményeket.
- `chunk_size` (`--chunk_size`, option, alapértelmezés: `5.0`): Feldolgozási chunk hossza percben. `0` esetén a teljes fájlt egyben dolgozza fel.
- `chunk_overlap` (`--chunk_overlap`, option, alapértelmezés: `10.0`): Chunk átfedés másodpercben. Nagyobb érték simább átmenetet ad, de lassítja a feldolgozást.
- `non_speech_silence` (`--non_speech_silence`, flag, alapértelmezés: `false`): Ha be van kapcsolva, a non-speech sáv csak csendet tartalmaz; így a háttér hang nélkül menthető.
- `background_blend` (`--background_blend`, option, alapértelmezés: `0.5`): 0–1 közötti érték, amely szabályozza, mennyire keverje vissza az eredeti mixet a modell által becsült háttérrel.
- `keep_full_audio` (`--keep_full_audio`, flag, alapértelmezés: `false`): Elmenti a konvertált teljes audiót is a `separated_audio_speech` mappába.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás a `tools.debug_utils` modulon keresztül.
