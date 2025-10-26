# merge_chunks_with_background_easy – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `merge_chunks_with_background_easy.py`

Ez a szkript a `translated_splits` mappában található generált szegmenseket időbélyegeik alapján összefűzi, és – igény szerint – rákeveri a háttér audiót. A végeredményt a `film_dubbing` könyvtárba menti.

## Kötelező beállítás
- `project_name` (pozícionális, alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális beállítások
- `narrator` (`-narrator`, `--narrator`, flag, alapértelmezés: `false`): Speciális „narrátor” üzemmód. Ilyenkor a háttérsáv a `extracted_audio` könyvtárból kerül kiválasztásra, és a szkript feltételezi, hogy narráció készül.
- `background_volume` (`--background-volume`, option, alapértelmezés: `100`): Narrátor módban a háttér sáv hangereje százalékban (1–100). Ha nincs narrátor mód, a szkript hibaüzenettel jelzi, hogy ez az opció nem érvényes.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletes naplózást kapcsol be (ha a `tools.debug_utils` elérhető).
