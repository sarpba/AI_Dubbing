# collect_normalized_translations – konfigurációs útmutató

**Futtatási környezet:** `f5-tts`  
**Belépési pont:** `collect_normalized_translations.py`

A szkript bejárja a projekt `translated` könyvtárát, begyűjti az összes `translated_text` mezőt, lefuttatja rájuk a magyar normalizálót, majd naplófájlba menti az eredményt. Emellett gondoskodik a `config.json` alapú projektútvonalak felderítéséről.

## Kötelező beállítás
- `project_dir` (`-p`, `--project`, option, alapértelmezés: nincs): A projekt neve vagy útvonala. Ha relatív nevet adsz meg, a szkript a `config.json` alapján keresi a `workdir` alatt.

## Opcionális beállítások
Jelenleg nincs további parancssori kapcsoló; a szkript minden egyéb viselkedést a konfigurációs fájl alapján határoz meg.
