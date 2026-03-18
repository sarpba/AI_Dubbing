# Belső Leírás: Új Script Modul Fejlesztése

Ez a leírás nem végfelhasználói dokumentáció, hanem belső emlékeztető arról, hogyan kell ebbe a keretrendszerbe olyan új scriptet tenni, ami elsőre felismerhető, futtatható és UI-ból kezelhető.

A cél nem egy "szép" script, hanem egy olyan modul, ami:
- megjelenik a webes workflow editorban,
- helyesen kapja meg a paramétereit,
- működik kézi CLI futtatással is,
- helpet ad a felületen,
- nem igényel utólagos kézi hackelést a központi registry-ben.

## 1. Kötelező fájlhármas

Minden új modul három, azonos alaptörzsű fájlból álljon ugyanabban a mappában:

1. `<name>.py`
2. `<name>.json`
3. `<name>_help.md`

Példa:

```text
scripts/TRANSLATE/my_tool/
  my_tool.py
  my_tool.json
  my_tool_help.md
```

Fontos:
- a `.json` csak akkor kerül be a script katalógusba, ha mellette ugyanazzal a névvel létezik a `.py` is
- a help fájl neve pontosan `<stem>_help.md` legyen
- a `scripts/scripts.json` automatikusan újraépül a `scripts/**/**/*.json` fájlokból, ezért azt nem kézzel kell szerkeszteni

## 2. Hová tedd a scriptet

A scriptet mindig a `scripts/` alatt helyezd el, valamilyen logikus kategóriában.

Példák:
- `scripts/ASR/...`
- `scripts/TRANSLATE/...`
- `scripts/TTS/...`
- `scripts/AUDIO-VIDEO/...`
- `scripts/TEST_SCRIPTS/...`

Teszt script lehet a repo-ban, a rendszer ezt kezeli. A fontos az, hogy a modul fájlszerkezete és metaadata helyes legyen.

## 3. A Python script minimális szerkezete

Az új script legyen önmagában is futtatható CLI program.

Minimum elvárás:
- `argparse` alapú CLI
- `main()` függvény
- `if __name__ == "__main__": main()`
- projektfüggő működésnél legyen projekt paraméter
- `config.json` alapján dolgozzon, ne beégetett abszolút útvonalakkal
- értelmes hibaüzenettel álljon le
- idempotens legyen, vagy legalább ne roncsolja a korábbi kimenetet

Ajánlott közös debug minta:

```python
from tools.debug_utils import add_debug_argument, configure_debug_mode

parser = argparse.ArgumentParser(...)
add_debug_argument(parser)
args = parser.parse_args()
log_level = configure_debug_mode(args.debug)
logging.basicConfig(level=log_level, ...)
```

Ez nem minden régi scriptben következetes, de az új modulnál ezt kell követni.

## 4. Projektfüggő path kezelés

Ne használj beégetett útvonalakat.

Az elvárt működés:
- a script találja meg a repo gyökerét
- onnan olvassa a `config.json`-t
- a projekt neve alapján a `workdir` és a projekt almappák ebből legyenek feloldva

Gyakorlati szabály:
- ha a script projektre dolgozik, legyen valamilyen `project_name` vagy azzal ekvivalens paramétere
- a kimeneteket a projekt struktúráján belül hozza létre

## 5. A JSON descriptor valós sémája

Az új `.json` fájl ténylegesen ezt a struktúrát kövesse:

```json
{
  "environment": "sync",
  "script": "CATEGORY/module/name.py",
  "description": "Rövid, egy mondatos leírás.",
  "required": [],
  "optional": [],
  "api": "chatgpt"
}
```

Megjegyzések:
- új fájlban `environment` kulcsot használj, ne `enviroment`
- a loader jelenleg mindkettőt elfogadja, de az új helyes alak az `environment`
- a `script` mező a `scripts/` alatti relatív útvonal legyen
- ennek egyeznie kell a valódi `.py` fájl helyével
- a `description` rövid legyen, mert ez jelenik meg a felületen is
- az `api` opcionális

## 6. Paramétertípusok és azok jelentése

A backend jelenleg ezeket a `type` értékeket kezeli:

- `flag`
- `option`
- `positional`
- `config_option`

Jelentésük:

### `flag`
Boolean kapcsoló.

Példa:

```json
{
  "name": "debug",
  "flags": ["--debug"],
  "type": "flag",
  "default": false
}
```

CLI oldalon ez tipikusan `action="store_true"` vagy hasonló.

### `option`
Kapcsoló + érték.

Példa:

```json
{
  "name": "project_name",
  "flags": ["-p", "--project-name"],
  "type": "option",
  "default": null
}
```

Ez a command build során így épül:

```text
--project-name demo_project
```

### `positional`
Pozicionális argumentum.

Példa:

```json
{
  "name": "input_file",
  "type": "positional",
  "default": null
}
```

Ehhez általában nincs `flags`.

### `config_option`
Speciális `name=value` formátum.

Példa:

```json
{
  "name": "temperature",
  "type": "config_option",
  "default": 0.7
}
```

Ez a command build során így lesz:

```text
temperature=0.7
```

## 7. Minden paraméterhez legyen explicit `default`

Ez fontos.

Minden paraméterobjektumban legyen `default`, akkor is, ha az érték `null`.

Példa:
- kötelező opcionális nélkül: `"default": null`
- bool: `"default": false` vagy `true`
- szám: `"default": 0.6`
- sztring: `"default": "gpt-4o"`

Ne hagyd el a `default` mezőt.

## 8. Required és optional paraméterek

A UI és a workflow editor a `required` és `optional` tömbökből épül fel.

Szabály:
- ami tényleg kötelező a futtatáshoz, menjen a `required` tömbbe
- minden más menjen az `optional` tömbbe
- a JSON, a `argparse` és a `_help.md` ugyanazokat a paramétereket tartalmazza

## 9. Flag-ek: pozitív és negatív logika

Ez a rendszer egyik kényes pontja. Új scriptnél ezt nagyon tudatosan kell megcsinálni.

### Normál pozitív flag

Példa:

```json
{
  "name": "stream",
  "flags": ["--stream"],
  "type": "flag",
  "default": false
}
```

Jelentés:
- alapból ki van kapcsolva
- ha a user bekapcsolja, a parancsba bekerül a `--stream`

### Pozitív és negatív pár

Példa:

```json
{
  "name": "backup",
  "flags": ["--backup", "--no-backup"],
  "type": "flag",
  "default": true
}
```

Jelentés:
- alapból bekapcsolt funkció
- ha a user kikapcsolja, a backend a `--no-backup` flaget használja

### Csak negatív flag

A rendszer kezeli azt is, amikor a CLI-ben csak `--no-*` kapcsoló van.

Ilyenkor a paraméter nevét úgy kell megválasztani, hogy a UI értelmesen tudja megjeleníteni.

Ajánlott minták:
- `no_backup`
- `disable_cache`
- `skip_cleanup`
- `without_diarization`

Példa:

```json
{
  "name": "no_backup",
  "flags": ["--no-backup"],
  "type": "flag",
  "default": false
}
```

Ennek a jelenlegi UI-logikában az a jelentése:
- a funkció alapból be van kapcsolva
- a jelölő a tényleges funkcióállapotot mutatja
- kikapcsoláskor a backend a negatív flaget adja át

Fontos:
- ha csak negatív flag van, de a `name` nem ilyen prefixszel kezdődik, a UI könnyen félreérthető lesz
- támogatott negatív prefixek jelenleg:
  - `no_`
  - `disable_`
  - `skip_`
  - `without_`

Új scriptnél ezt a mintát kövesd.

## 10. Titkos kulcsok és keyholder integráció

Nem elég az, hogy a scriptnek van például `auth_key` paramétere. A webes rendszer csak akkor tud vele jól bánni, ha a backend mappingjei is helyesek.

Jelenlegi általános secret paraméternevek:
- `auth_key`
- `api_key`
- `hf_token`

Ha az új script ilyen paramétert használ:
- a workflow UI maszkolni fogja az értéket
- a mentett workflow-ban kódolt formában maradhat meg

Ha azt akarod, hogy a rendszer automatikusan a keyholderből töltse fel:
- fel kell venni a scriptet a [main_app.py](/home/sarpba/Sajat_programok/AI_Dubbing/main_app.py#L563) `SCRIPT_PARAM_KEYHOLDER` mappingjébe

Ha azt akarod, hogy a workflow előre jelezze a hiányzó API kulcsot:
- fel kell venni a scriptet a [main_app.py](/home/sarpba/Sajat_programok/AI_Dubbing/main_app.py#L556) `SCRIPT_KEY_REQUIREMENTS` mappingjébe

Összefoglalva:
- csak a JSON önmagában nem elég a kulcs-integrációhoz
- ha új külső szolgáltatás jön be, a backend mappinget is frissíteni kell

## 11. Projekt paraméterek automatikus kitöltése

A workflow editor bizonyos paramétereket automatikusan fel tud tölteni a projektből.

Biztonságos, ajánlott nevek:
- `project_name`
- `project`
- `project_dir_name`
- `project_dir`
- `project_path`

Jelenlegi backend logika:
- ha a név projektnevet jelent, a rendszer a projekt nevét tudja átadni
- ha a név útvonalat jelent, a projekt teljes útvonalát tudja átadni

Ha lehet, új scriptnél ezeket a kanonikus neveket használd.

## 12. Help fájl szerepe

A `<name>_help.md` a felületen help modalként jelenik meg. Ez nem opcionális díszítés, hanem a UI része.

A help legyen:
- rövid
- pontos
- a JSON defaultjaival összhangban
- különösen egyértelmű a kapcsolóknál

Ajánlott szerkezet:

```md
# script_nev

**Futtatási környezet:** `sync`
**Belépési pont:** `CATEGORY/module/name.py`

## Mit csinál?
Rövid leírás.

## Kötelező paraméterek
- ...

## Opcionális paraméterek
- ...

## Megjegyzés
- ha van negatív flag vagy más különleges viselkedés, itt írd le
```

Ne írj bele olyat, ami nincs a JSON-ban vagy a CLI-ben.

## 13. A script katalogizálás valós működése

A rendszer a script listát a `scripts/` mappából építi újra.

Fontos működés:
- a `scripts/scripts.json` derivált fájl
- a forrás az egyes modulok saját `.json` fájlja
- a rendszer a `.json`-t beolvassa, validálja, és a valós `.py` elérési út alapján újraépíti a közös listát

Tehát:
- ne kézzel szerkeszd a `scripts/scripts.json`-t
- az egyedi modul `.json` fájlját szerkeszd

## 14. Amit a backend ténylegesen elvár

Új scriptnél ezek legyenek igazak:

- a `script` mező helyes relatív útvonal a `scripts/` alól
- az `environment` valós conda környezetet jelöl
- minden paraméternek van `name`
- minden paraméternek van `type`
- minden paraméternek van `default`
- `flag` típusnál a `flags` logikája megfelel a valódi CLI-nek
- a `_help.md` létezik és olvasható
- a `.py` tényleg ott van, ahová a JSON mutat

Ha bármelyik eltér, a modul vagy nem jelenik meg jól, vagy hibásan fut.

## 15. Ajánlott minimál minta új projektfüggő scripthez

Példa JSON:

```json
{
  "environment": "sync",
  "script": "MY_CATEGORY/example/example_tool.py",
  "description": "Példa script egy projektfájl feldolgozására.",
  "required": [
    {
      "name": "project_name",
      "flags": ["-p", "--project-name"],
      "type": "option",
      "default": null
    }
  ],
  "optional": [
    {
      "name": "output_suffix",
      "flags": ["--output-suffix"],
      "type": "option",
      "default": "_processed"
    },
    {
      "name": "debug",
      "flags": ["--debug"],
      "type": "flag",
      "default": false
    }
  ]
}
```

Ez jó alap, mert:
- van projekt paraméter
- van explicit default
- a debug standard módon működik
- a UI gond nélkül rendereli

## 16. Ellenőrzési lista új modul hozzáadása után

Legalább ezeket ellenőrizd:

- létezik a `.py`, `.json`, `_help.md` fájlhármas
- a három fájl alaptörzse azonos
- a `script` mező helyes relatív útvonal
- az `environment` kulcs helyesen van írva
- minden paraméterben van `default`
- a CLI és a JSON argumentumnevei egyeznek
- a help fájl ugyanezeket a paramétereket írja le
- a negatív flag logika nem félreérthető
- ha secret kell, a backend mappingek is frissültek

## 17. Gyakorlati validálás új modul után

Ezt csináld:

1. futtasd kézzel a scriptet `--help`-pel
2. ellenőrizd, hogy a modul megjelenik a workflow editorban
3. nyisd meg a help modalt
4. nézd meg, hogy a default értékek a UI-ban helyesek-e
5. ha van flag, ellenőrizd a kiinduló kapcsolóállapotot
6. ha van secret paraméter, ellenőrizd a keyholder integrációt

Repo szinten érdemes lefuttatni:

```bash
python tools/validate_script_meta.py
python -m unittest discover -s tests -v
```

## 18. Mit ne csinálj

- ne szerkeszd kézzel a `scripts/scripts.json`-t
- ne használj új scriptben `enviroment` kulcsot
- ne hagyd le a `default` mezőket
- ne adj a JSON-ba olyan paramétert, ami nincs a CLI-ben
- ne hagyd, hogy a help fájl eltérjen a JSON-tól
- ne használj félrevezető negatív flag-nevet
- ne hardcode-olj lokális abszolút pathokat

## 19. Rövid döntési szabályok

Ha gyorsan kell eldönteni, hogy egy új script "keretrendszer-kompatibilis-e", ez legyen a rövid szűrő:

- van `.py` + `.json` + `_help.md`
- a JSON-ban `environment`, `script`, `description`, `required`, `optional` rendben van
- minden paraméter kapott `default`-ot
- a CLI és a JSON 1:1-ben megfelel egymásnak
- a projektparaméter neve kanonikus
- a flag logika megfelel a tényleges alapállapotnak
- a help fájl rövid és pontos
- ha kell API kulcs, a backend mapping is frissült

Ha ez mind teljesül, az új modul nagy valószínűséggel elsőre beilleszthető lesz.
