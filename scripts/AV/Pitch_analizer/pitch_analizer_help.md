# pitch_analizer – használati útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `AV/Pitch_analizer/pitch_analizer.py`

A szkript a narrátor referencia audió medián hangmagasságát hasonlítja össze a projekt `translated_splits` mappájában található kimeneti fájlokkal. Az eltérést hertzben méri, és jelzi, ha bármelyik szegmens túllépi az engedélyezett toleranciát.

## Kötelező beállítások
- `project_name` (`-p`, `--project-name`, option, alapértelmezés: nincs): A `workdir` alatti projekt azonosítója, amelynek `translated_splits` könyvtárát vizsgáljuk.
- `narrator` (`--narrator`, option, alapértelmezés: nincs): A narrátor referencia audiót tartalmazó könyvtár elérési útja. A mappában legalább egy `.wav`, `.flac` vagy `.ogg` fájlnak kell lennie; az első megfelelőt használja baseline-ként.

## Opcionális beállítások
- `tolerance` (`--tolerance`, option, alapértelmezés: `20.0`): Megengedett hangmagasság-eltérés hertzben. Ennél nagyobb differencia riasztásként jelenik meg.
- `min_frequency` (`--min-frequency`, option, alapértelmezés: `60.0`): A pitch-keresés alsó határa (Hz); mély férfihangnál érdemes csökkenteni.
- `max_frequency` (`--max-frequency`, option, alapértelmezés: `400.0`): A pitch-keresés felső határa (Hz); magas narrátor hang esetén növelhető.
- `delete_outside` (`--delete-outside`, flag, alapértelmezés: `false`): Ha engedélyezed, a tolerancián kívül eső fájlokat a szkript automatikusan törli a vizsgálat végén.
- `debug` (`--debug`, flag, alapértelmezés: `false`): Részletesebb logokat és UserWarning üzeneteket kapcsol be.

## Folyamat áttekintése
1. Betölti a projekt gyökeréből a `config.json`-t, majd feloldja a `workdir/<project_name>/translated_splits` elérési utat.
2. A narrátor mappából beolvassa az első támogatott audió fájlt, mono jellé alakítja, normalizálja, és kiszámítja a medián hangmagasságot.
3. A CPU magok felét (legkevesebb 1 worker) felhasználva párhuzamosan elemzi a `translated_splits` könyvtár támogatott audió fájljait, mindegyikre hangmagasság-becslést végez (autokorrelációs módszerrel).
4. Minden fájlhoz kiírja a medián Hz értéket, a narrátortól való eltérést, valamint a hangos (voiced) keretek arányát.
5. A tolerancián kívül eső fájlokat külön listázza (és opcionálisan törli), a folyamat végén nem nulla kilépési kóddal tér vissza (`2`), így automatizmusban is könnyen ellenőrizhető.

## Hasznos tippek
- A szkript a `numpy` és `soundfile` csomagokat igényli; ha hiányoznak, telepítsd őket a használt környezetbe (`pip install numpy soundfile`).
- A narrátor könyvtárban érdemes csak egy végleges referencia fájlt tartani, így nem jelenik meg figyelmeztetés többszörös találat miatt.
- Ha túl sok szegmens kerül „KINT” státuszba, próbálj magasabb toleranciát vagy szélesebb frekvenciatartományt megadni, illetve ellenőrizd, hogy a narrátor referencia és a szintetizált hangok hasonló mintavételi frekvenciával készültek-e.
- Automatizált futtatásoknál a kilépési kód alapján dönthetsz: `0` minden rendben, `1` konfigurációs/IO hiba, `2` legalább egy szegmens túllépi az engedélyezett eltérést.
