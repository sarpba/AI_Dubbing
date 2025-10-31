# Útmutató új szkript modul létrehozásához

Ez a dokumentum egyben foglalja a kód (`.py`), a hozzá tartozó konfiguráció (`.json`) és a felhasználói leírás (`_help.md`) elkészítésének lépéseit. **Minden új modulhoz mindhárom fájlt kötelező létrehozni**, különben a rendszer vagy a felület hiányos információból dolgozik.

Az itt leírtak a korábbi részdokumentációk tartalmát egyesítik; a példákhoz és további inspirációhoz nézd meg a meglévő modulokat.

## Alapelvek

- **Konfiguráció-központúság** – Minden projekt-specifikus mappát a gyökér `config.json` alapján oldj fel (`workdir/<projekt_név>/...`), ne használj hardkódolt elérési utat.
- **Projekt-paraméter** – A legtöbb modul `-p/--project-name` kapcsolóval kapja meg, melyik projektet kell feldolgozni.
- **Egységes CLI leírás** – A modulhoz tartozó `<név>.json` definiálja az argumentumokat; ezeknek egyezniük kell az argparse beállításokkal. A központi `scripts/scripts.json` állományt nem kell kézzel szerkeszteni, a fő alkalmazás frissíti.
- **Debug támogatás** – Mindig vedd fel az `add_debug_argument` és `configure_debug_mode` hívásokat, hogy a naplózás konzisztens legyen.
- **Idempotens működés** – Többszöri futtatás ne generáljon duplikált vagy sérült kimenetet; ahol kell, ellenőrizd, hogy a célfájl már létezik-e.

---

## 1. Tervezés – követelmények összegyűjtése

1. **Feladat meghatározása** – röviden rögzítsd, mit csinál a szkript, milyen bemenetet használ, mi az elvárt kimenet.
2. **Projektstruktúra** – nézd meg a `config.json`-t, hogy mely almappákat kell olvasni vagy írni (pl. `separated_audio_speech`, `translated`, `film_dubbing`).
3. **Függőségek** – döntsd el, szükség van-e külső csomagra vagy binárisra (pl. `ffmpeg`, `deepl`, `torch`). Ha igen, gondoskodj hibatűrő ellenőrzésről.

---

## 2. A Python szkript (`.py`)

### Kötelező elemek

- **Projektgyökér feloldása**  
  Használd a bevált `get_project_root()` sablont (lásd meglévő szkriptek), amely a fához közelebbi `config.json` alapján határozza meg a gyökeret.

- **Konfiguráció betöltése**  
  Írj `load_config()` vagy hasonló segédfüggvényt, amely JSON-t olvas, hibát jelez, és visszaadja a konfigurációt és/vagy a project gyökerét.

- **Argparse definíció**  
  - Kötelező legalább egy projekt azonosító argumentum (`-p/--project-name`, vagy típusának megfelelő név).
  - Hívd meg a `add_debug_argument(parser)` függvényt; a futás elején pedig `configure_debug_mode(args.debug)`-et.
  - Minden CLI opcióhoz adj típusinformációt (`type=int`, `type=float` stb.) és alapértéket, ha van.

- **Útvonalak feloldása**  
  A `config["DIRECTORIES"]` és `config["PROJECT_SUBDIRS"]` kulcsok használatával számítsd ki a bemeneti és kimeneti mappákat. Ellenőrizd, hogy léteznek, és barátságos hibaüzenettel állj le, ha nem.

- **Core logika**  
  - A moduloktól elvárt, hogy idempotensek legyenek (többszöri futtatás ne vezessen váratlan eredményhez).
  - A műveletekről írj rövid logot (`print` vagy `logging`), különösen a mentett fájlok nevéről.

- **Fő belépési pont**  
  Tartsd meg a szokásos szerkezetet:
  ```python
  def main() -> None:
      parser = argparse.ArgumentParser(...)
      add_debug_argument(parser)
      args = parser.parse_args()
      log_level = configure_debug_mode(args.debug)
      logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
      ...

  if __name__ == "__main__":
      main()
  ```

### Ajánlott minták

| Feladat | Példa | Mit érdemes figyelni |
| --- | --- | --- |
| Audio feldolgozás | `AUDIO-VIDEO/merge_chunks_with_background/merge_chunks_with_background_easy.py` | Konfigurálható mappanevek, háttér hangerő | 
| API hívás | `TRANSLATE/deepl/deepl_translate.py` | Kulcskezelés (`keyholder.json`), opcionális paraméterek | 
| Többlépcsős adatfeldolgozás | `ASR/resegment/resegment.py` | Biztonsági mentés, opcionális finomhangoló lépések |

### Ajánlott gyakorlatok

- Használj típusannotációkat (`Optional[str]`, `Tuple[...]`) a könnyebb karbantarthatóság érdekében.
- JSON mentéskor állítsd `ensure_ascii=False`-ra, hogy az ékezetek megmaradjanak.
- Külső függőség hiányát kezeld barátságos hibával és konkrét telepítési javaslattal.
- Logold az induló beállításokat és minden létrehozott/out fájlt („Saved …”).
- Hibakezeléskor ne hagyd, hogy nyers kivétel jusson a felhasználóig; fogd meg, logold, majd `sys.exit(1)`.

---

## 3. A konfigurációs leíró (`.json`)

Minden szkriptnek saját, azonos nevű JSON fájlra van szüksége (pl. `resegment.py` → `resegment.json`). Ez írja le a felületnek és az automatizmusoknak, milyen kapcsolók érhetők el.

### Kötelező mezők

- `enviroment`: a futtatáshoz használt virtuális környezet neve (pl. `sync`, `nemo`, `f5-tts`).
- `script`: a Python fájl relatív útja (pl. `ASR/resegment/resegment.py`).
- `description`: rövid, természetes nyelvű összefoglaló.
- `required`: lista a kötelező argumentumokról.

### Opcionális mezők

- `api`: ha külső szolgáltatás kulcsára van szükség (pl. `deepl`, `huggingface`, `chatgpt`).
- `optional`: lista az opcionális argumentumokról.

### Paraméter leírási séma

```json
{
  "name": "project_name",
  "flags": ["-p", "--project-name"],
  "type": "option",
  "default": null
}
```

- `name`: belső azonosító; érdemes az argparse-ban használt célváltozóval egyezőnek lennie.
- `flags`: a CLI kapcsolók; ha hiányzik, a paraméter pozícionális.
- `type`: `option`, `flag`, `positional` vagy `config_option`.
- `default`: JSON-kompatibilis alapérték (`null`, `true`, `false`, szám, string).

### Tippek a konzisztenciához

1. A JSON pontosan egyezzen az argparse definícióval (név, flag, típus, default).
2. Boolean flag-nél a `default` tükrözze a kezdeti állapotot.
3. Ha ellentétes flagpárokat kínálsz (pl. `--feature` / `--no-feature`), mindkettőt rögzítsd.
4. A `script` mező legyen mindig a valós relatív út.
5. A `default` érték legyen JSON-kompatibilis (szám, string, bool vagy `null`).
6. Az `api` mező segítségével jelezd, ha a modul külső kulcsot használ (pl. `keyholder.json`).
7. A központi `scripts/scripts.json`-t nem szükséges frissítened; a `main_app` indításakor automatikusan újragenerálja a bejegyzéseket.

---

## 4. A felhasználói leírás (`_help.md`)

Ez a markdown fájl szolgál a dokumentációra és a kezelőfelületen megjelenő súgóra. A név formátuma legyen `<script>_help.md` (pl. `resegment_help.md`).

### Tartalomjegyzék javaslat

1. **Cím** – a szkript neve és rövid tag-line.
2. **Futtatási környezet / belépési pont** – adja meg az `enviroment` és `script` értékét.
3. **Áttekintés** – mire való, rövid workflow, előfeltételek.
4. **Kötelező beállítások** – sorold fel a `required` paramétereket, jelezve a flag-eket.
5. **Opcionális beállítások** – csoportosítva, rövid magyarázattal és alapérték megadásával.
6. **Kimenet / melléktermékek** – milyen fájlokat hoz létre, hol.
7. **Hibakezelés / tippek** – tipikus problémák, javasolt eljárás (pl. hiányzó audio, API kulcs).

### Írási irányelvek

- Használj magyar nyelvű, tömör, de részletes mondatokat.
- Ha kódot vagy parancsot mutatsz be, használj ``` code blockot.
- Tüntesd fel az alapértelmezéseket és a paraméterek kapcsolatát (`--no-backup` kapcsolja ki a `--backup` által vezérelt funkciót).
- Jelezd, ha valamelyik paraméter csak első futtatásnál kötelező (pl. API kulcs megadása).

---

## 5. Gyors ellenőrzési lista

1. **Fájlstruktúra**
   - [ ] `scripts/<útvonal>/<név>.py`
   - [ ] `scripts/<útvonal>/<név>.json`
   - [ ] `scripts/<útvonal>/<név>_help.md`

2. **CLI ↔ JSON ↔ Dokumentáció szinkron**
   - [ ] Az argparse definíció megegyezik a JSON paraméterlistával.
   - [ ] A help fájl minden kötelező és opcionális paramétert felsorol azonos névvel.

3. **Konfigurációs hivatkozások**
   - [ ] A szkript a `config.json`-ból oldja fel az elérési utakat.
   - [ ] Hibás konfiguráció esetén egyértelmű üzenettel áll le.

4. **Logging és debug**
   - [ ] `add_debug_argument` meghívva.
   - [ ] `configure_debug_mode` beállítja a log szintet.
   - [ ] A fontos műveletek (fájl olvasás/írás, API hívás) logolva vannak.

5. **Robusztusság**
   - [ ] Kezeli a hiányzó bemeneti fájlokat / üres mappákat.
   - [ ] Több futtatásnál nem generál duplikált vagy sérült kimenetet.
   - [ ] Külső függőség hiányát felhasználóbarát hibaüzenet jelzi.
   - [ ] API kulcsot igénylő modulnál megoldott a `keyholder.json`-ba mentés/betöltés.
   - [ ] `python <relatív út>/<név>.py --help` kimenete egyezik a dokumentációval.

---

## 6. Fejlesztői tippek

- **Verziókezelés** – minden új modulhoz külön commitot készíts, így könnyebb visszakövetni a változásokat.
- **Tesztfuttatás** – még dokumentáció frissítés után is futtasd a `python <script> --help` parancsot, hogy biztosan egyezzen a JSON-nal és a súgóval.
- **Meglévő modulok újrahasznosítása** – ha hasonló feladatot valósítasz meg (pl. audio mux, TTS), másold át a bevált segédfüggvényeket, majd alakítsd a saját igényeidre.
- **Kulcskezelés** – API kulcsot igénylő moduloknál használd a `keyholder.json`-t, és mindig kódolj base64-lel a tároláshoz.
- **Nem ASCII karakterek** – JSON mentéskor `ensure_ascii=False` beállítással megőrizhetők az ékezetek; a Python fájlban is használj UTF-8-at.
- **Naplózás** – induláskor írd ki a kulcs paramétereket (projekt, bemenet, kimenet), így egyszerűbb a hibakeresés.
- **Mintakódok** – nézd át az ASR, TRANSLATE, AUDIO-VIDEO vagy TTS almappák meglévő megoldásait inspirációért.

---

## 7. Összegzés

Egy új szkript modul akkor tekinthető „késznek”, ha:

1. A Python fájl a projektstruktúrára támaszkodik és jó hibatűrésű.
2. A hozzátartozó JSON pontosan leírja az elérhető paramétereket.
3. A `_help.md` dokumentum a felhasználó szemszögéből teljes képet ad – beleértve az előfeltételeket, futtatási példát és a kimenetet.

Az egységes keretrendszer előnye, hogy a front-end, a CLI és az automatizmusok is azonos információból dolgoznak, így a fejlesztői és felhasználói élmény kiszámítható marad. Tartsd a fenti checklisteket kéznél minden új modulnál!

---

👉 **További források:**  
- Meglévő minták a `scripts/` könyvtárban (ASR, TRANSLATE, AUDIO-VIDEO, TTS almappák)  
- [scripts/how_to_create_script_modul_AI.md](how_to_create_script_modul_AI.md) – tömör, angol nyelvű instrukció AI ágensek számára
