# EQ.json – konfigurációs útmutató

Ez a fájl egy többpontos EQ görbét ír le, amelyet az F5-TTS szkriptek a referencia audió előfeldolgozásakor tudnak alkalmazni (`--eq-config`). A struktúra teljesen önálló; más szkriptek is felhasználhatják, ha ugyanilyen formátumot várnak.

## Mezők
- `description`: Szabad szöveges leírás az EQ céljáról. Dokumentációs célt szolgál, a szkript nem használja programozottan.
- `global_gain_db`: Teljes spektrumra érvényes hangerőkorrekció decibelben. Pozitív érték emel, negatív csökkent.
- `points`: Rendezett lista, amely az EQ töréspontjait tartalmazza.
  - `frequency_hz`: A kontrollpont frekvenciája Hertzben. A pontok sorrendje számít, a szkript szükség esetén rendezi.
  - `gain_db`: A megadott frekvencián alkalmazott erősítés/csillapítás decibelben. A köztes értékeket a szkript interpolálja.

## Használati javaslatok
- A `points` sorozat lefedi a teljes hallható tartományt 0–24 kHz között, így alapértelmezettnek is használható.
- Saját EQ görbéhez elegendő a listában szereplő pontokat módosítani. Ügyelj arra, hogy a frekvencia ne csökkenjen 0 alá, és hogy minden ponthoz tartozzon `gain_db`.
- Az F5-TTS szkriptek a normalizálás után alkalmazzák a görbét, ezért a `global_gain_db` értékét úgy állítsd be, hogy a referencia ne lépje túl a kívánt csúcsot (pl. együtt a `--ref-audio-peak` beállítással).
