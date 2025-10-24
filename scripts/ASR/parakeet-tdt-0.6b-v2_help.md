# parakeet-tdt-0.6b-v2 – konfigurációs útmutató

**Futtatási környezet:** `nemo`  
**Belépési pont:** `ASR/parakeet-tdt-0.6b-v2.py`

A szkript az NVIDIA Parakeet TDT 0.6B V2 modellt futtatja, automatikus vagy kézi darabolással, és időbélyeges JSON átiratokat készít a `separated_audio_speech` mappában található hangokról.

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir`-ben található projektmappa neve, amelynek hanganyagát feldolgozza.

## Opcionális beállítások
- `no_auto_chunk` (`--no-auto-chunk`, flag, alapértelmezés: `false`): Kikapcsolja az automatikus chunk-méret kalibrációt. Ha aktiválod, a `--chunk` értéke lesz érvényes.
- `chunk` (`--chunk`, option, alapértelmezés: `30`): Fix chunkhossz másodpercben, amikor az automatikus kalibráció ki van kapcsolva.
- `max_pause` (`--max-pause`, option, alapértelmezés: `0.6`): A mondatok közti maximális szünet másodpercben; nagyobb érték kevésbé darabolja szét a hosszú szüneteket.
- `timestamp_padding` (`--timestamp-padding`, option, alapértelmezés: `0.2`): Az időbélyegekhez hozzáadott extra margó másodpercben, hogy a szegmensek ne vágják le a beszéd elejét-végét.
- `max_segment_duration` (`--max-segment-duration`, option, alapértelmezés: `11.5`): Egy szegmens maximális hossza. `0` értékkel kikapcsolható a korlátozás.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózás bekapcsolása.
