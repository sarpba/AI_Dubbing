# parakeet-tdt-0.6b-v3 – konfigurációs útmutató

**Futtatási környezet:** `nemo`  
**Belépési pont:** `parakeet-tdt-0.6b-v3.py`

Ez a szkript a Parakeet TDT 0.6B V3 modellre épít, és JSON alapú átiratokat készít a projekt hanganyagaiból. A V2 verzióhoz hasonlóan támogatja az automatikus és a kézi darabolást.

## Kötelező beállítás
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A feldolgozandó projekt könyvtárának neve a `workdir` alatt.

## Opcionális beállítások
- `no_auto_chunk` (`--no-auto-chunk`, flag, alapértelmezés: `false`): Kikapcsolja a dinamikus chunk méretszabályozást, így a `--chunk` fix értéke lesz érvényes.
- `chunk` (`--chunk`, option, alapértelmezés: `30`): Fix chunkhossz másodpercben, ha manuálisan akarod szabályozni a feldolgozást.
- `max_pause` (`--max-pause`, option, alapértelmezés: `0.6`): Maximális szünet hossza a mondatok között másodpercben; nagyobb érték visszafogja a darabolást.
- `timestamp_padding` (`--timestamp-padding`, option, alapértelmezés: `0.2`): Az időbélyegekhez hozzáadott extra margó másodpercben a beszéd elejének-végének megőrzéséhez.
- `max_segment_duration` (`--max-segment-duration`, option, alapértelmezés: `11.5`): Egy szegmens felső hosszkorlátja másodpercben (`0` = nincs limit).
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózást kér a futás során.
