# merge_chunks_with_background_easy

**Futtatási környezet:** `sync`  
**Belépési pont:** `AUDIO-VIDEO/merge_chunks_with_bacground_v1.1/merge_chunks_with_background_easy.py`

## Mit csinál?
A korábban generált szinkron darabokat összemuxolja a háttér audióval. Képes automatikusan felgyorsítani az egymásra lógó hangrészeket.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project_name` (pozicionális;  kapcsoló: pozicionális; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `narrator` (kapcsoló;  kapcsoló: `-narrator`, `--narrator`; alapértelmezés: `false`): Narrátor módot használ, vagy megadja a narrátor referenciafájlt / mappát az adott script elvárásai szerint. Alapállapotban ki van kapcsolva.
- `background_volume` (opció;  kapcsoló: `--background-volume`; alapértelmezés: nincs): A háttérhang vagy zene hangerőszintje a keverés során.
- `time_stretching` (kapcsoló;  kapcsoló: `--time-stretching`; alapértelmezés: `false`): A script megengedett határok között gyorsítja vagy nyújtja a hangot, ha az időzítés ezt igényli. Alapállapotban ki van kapcsolva.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
