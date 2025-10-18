# Projekt-alapú scripts rendszer – útmutató

Ez a dokumentum összefoglalja, hogyan működik az AI_Dubbing projekt `scripts/` könyvtárában található projekt‑alapú szkriptrendszer, és mit érdemes követni új, config‑vezérelt szkriptek készítésekor.

## Alapelvek

- **Konfiguráció-központúság** – A szkriptek nem fix elérési utakkal dolgoznak, hanem a gyökérben található `config.json` alapján oldják fel a projektmappákat (`workdir/<projekt_név>/...`).
- **Projekt-paraméter** – A legtöbb szkript `-p/--project-name` CLI argumentummal kapja meg, melyik projektet kell feldolgozni.
- **Egységes CLI leírás** – A `scripts.json` (és az egyes szkriptekhez tartozó `<script>.json`) írja le, milyen paraméterekkel futtatható az adott szkript. Ezeket az UI és az automatizmusok is használják.
- **Debug támogatás** – Az `tools/debug_utils` modul `add_debug_argument` és `configure_debug_mode` függvényei közös logikát adnak a diagnosztikához.

## Kötelező lépések új projekt-alapú szkript írásakor

1. **Projekt gyökér meghatározása**
   ```python
   from pathlib import Path

   def get_project_root() -> Path:
       for candidate in Path(__file__).resolve().parents:
           if (candidate / "config.json").is_file():
               return candidate
       raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")
   ```

2. **config.json betöltése**
   ```python
   import json
   import sys

   def load_config() -> tuple[dict, Path]:
       project_root = get_project_root()
       config_path = project_root / "config.json"
       try:
           with open(config_path, "r", encoding="utf-8") as fp:
               config = json.load(fp)
           return config, project_root
       except (FileNotFoundError, json.JSONDecodeError) as exc:
           print(f"Hiba a konfiguráció betöltésekor ({config_path}): {exc}")
           sys.exit(1)
   ```

3. **Projekt-specifikus elérési utak feloldása**
   - Használd a `config["DIRECTORIES"]` és `config["PROJECT_SUBDIRS"]` mezőket.
   - Mindig ellenőrizd, hogy a számított mappák léteznek, és jelezd a hibát felhasználóbarát üzenettel.
   ```python
   def resolve_project_paths(project_name: str, config: dict, project_root: Path) -> Path:
       try:
           workdir = project_root / config["DIRECTORIES"]["workdir"]
           input_dir = workdir / project_name / config["PROJECT_SUBDIRS"]["separated_audio_speech"]
       except KeyError as exc:
           print(f"Hiba: hiányzó kulcs a config.json-ban: {exc}")
           sys.exit(1)
       if not input_dir.is_dir():
           print(f"Hiba: a feldolgozandó mappa nem található: {input_dir}")
           sys.exit(1)
       return input_dir
   ```

4. **CLI felület kialakítása**
   - Kötelező: `-p/--project-name`.
   - Szükség esetén opcionális paraméterek (pl. `--input-language`, `--output-language`), amelyek ha elhagyhatóak, akkor a `config.json` megfelelő alapértékeihez essenek vissza.
   - Mindig hívd meg az `add_debug_argument(parser)` és `configure_debug_mode(args.debug)` függvényeket.

5. **scripts.json és önálló JSON leíró frissítése**
   - A `scripts/scripts.json` listában add hozzá vagy módosítsd az adott szkript bejegyzését, hogy tükrözze az új CLI paramétereket.
   - Ha létezik külön leíró (`scripts/<ALMAPPÁK>/<script>.json`), tartsd szinkronban.

6. **Magyarázó naplózás**
   - Induláskor írasd ki a projektnevet és a használt könyvtárakat.
   - Ha a szkript sablonok, előfeldolgozások vagy kimenetek létrejöttét figyeli, szintén logold (pl. „Mentve: …”).

7. **Hibakezelés**
   - Bemeneti feltételeknél (pl. hiányzó JSON fájl, üres mappa) röviden, de egyértelműen írd ki a problémát.
   - Kerüld a kivétel visszadobását a felhasználói felületig – inkább `sys.exit(1)` vagy rendezett visszatérést használj a hibaüzenet után.

## Minták a meglévő szkriptekből

| Szkript | Kulcsfunkció | Megjegyzés |
| --- | --- | --- |
| `ASR/parakeet-tdt-0.6b-v2.py` | ASR feldolgozás `2_separated_audio_speech` mappából | Auto chunk, GPU kezelés, projekt név alapján dolgozik |
| `ASR/canary-easy.py` | Alternatív ASR pipeline | Szintén config‑vezérelt, több opcionális paraméterrel |
| `ASR/whisx.py` | WhisperX sok-GPU támogatással | A projekt könyvtáraiból dolgozik, `-p` paraméterrel |
| `TRANSLATE/translate.py` | DeepL fordítás | A config default nyelvi beállításait használja |
| `TRANSLATE/translate_chatgpt_srt_easy_codex.py` | ChatGPT-alapú fordítás + SRT igazítás | Részletes config betöltés, extra kulcskezelés (keyholder.json) |

Ezekből a kódokból könnyen kimásolhatóak a megoldások az útvonalkezelésre, CLI struktúrára vagy az API kulcsok kezelésére.

## Ajánlott gyakorlatok

- **Típuskövetelmények** – Használj `typing` típusannotációkat (`Optional[str]`, `Tuple[...]`) a könnyebb karbantarthatóság érdekében.
- **Unicode / JSON** – JSON írásakor `ensure_ascii=False` flaggel megőrizheted a magyar ékezeteket.
- **Függőség ellenőrzés** – Ha külső csomag kell (pl. `deepl`, `whisperx`), írd ki, ha hiányzik, és javasolj megoldást.
- **Idempotencia** – Törekedj arra, hogy a szkript ne generáljon duplikált kimeneteket, és szükség esetén ellenőrizze, létezik-e már az output.

## Gyors checklista fejlesztéskor

1. Betölti a `config.json`-t és megtalálja a projekt gyökerét?
2. `-p/--project-name` az elsődleges belépési pont?
3. Minden config-alapú mappa ellenőrzött és logolva van?
4. Van `debug` flag, amit a közös modul kezel?
5. Frissítve lett a `scripts.json` és a szkript saját JSON leírója?
6. Hiba esetén egyértelmű üzenettel leáll?
7. Újrafuttatható anélkül, hogy félbehagyná a pipeline-t?

Ezt az útmutatót tartsd referenciaként, amikor új projekt-alapú szkriptet írsz vagy meglévőt alakítasz át. A cél a konzisztens felhasználói élmény és az automatizálhatóság. Ha további példákra van szükséged, nézd meg a fenti mintaszkripteket. Kellemes fejlesztést! 💻
