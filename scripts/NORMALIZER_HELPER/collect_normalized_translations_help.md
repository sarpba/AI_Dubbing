# collect_normalized_translations – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `NORMALIZER_HELPER/collect_normalized_translations.py`

A szkript bejárja a projekt `translated` könyvtárát, begyűjti az összes `translated_text` mezőt, lefuttatja rájuk a magyar normalizálót, majd naplófájlba menti az eredményt. Emellett gondoskodik a `config.json` alapú projektútvonalak felderítéséről.

## Kötelező beállítás
- `project_dir` (`-p`, `--project`, option, alapértelmezés: nincs): A projekt neve vagy útvonala. Ha relatív nevet adsz meg, a szkript a `config.json` alapján keresi a `workdir` alatt.

## Opcionális beállítás
- `debug` (`--debug`, flag, alapértelmezés: `false`): A JSON metaadat tartalmazza, bár a jelenlegi Python verzió nem veszi figyelembe. Ha a futtatókörnyezet automatikusan hozzáadja a flaget, a szkript csendben ignorálja.
