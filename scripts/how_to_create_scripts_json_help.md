# scripts.json – konfigurációs útmutató

Ez a gyűjtő-fájl írja le, hogy az AI Dubbing felülete vagy CLI-je hogyan jelenítse meg és parametrizálja a scripts könyvtár parancsait. Minden elem egy önálló szkript-konfigurációt képvisel, amelyből a felület kiolvassa a felhasználónak kínált opciókat. A scripts.json-t ne frissítsd, csak a scriptel azonos nevű json filet hozd létre.

## Top-szintű mezők minden listatagban
- `enviroment`: A szkript futtatásához szükséges környezet/conda env neve. A launcher ennek megfelelően lép be a környezetbe.
- `api`: (Opcionális) Külső API azonosítója (`deepl`, `huggingface`, `chatgpt`, stb.), információs célra vagy jogosultságkezeléshez.
- `script`: A Python fájl relatív útja a `scripts` könyvtáron belül. Ez lesz a tényleges belépési pont.
- `description`: Emberi olvasóknak szánt összefoglaló arról, mit csinál a szkript.
- `required`: Lista a kötelező CLI paraméterekről. Minden elem mezői:
  - `name`: A belső paraméternév.
  - `flags`: (Opcionális) A CLI-n használható kapcsolók (pl. `["-p", "--project"]`). Ha hiányzik, pozícionális argumentumról van szó.
  - `type`: A paraméter típusa (`option`, `flag`, `positional`, `config_option`). A UI ebből tudja, hogy vár-e értéket.
  - `default`: Alapértelmezett érték. `null` esetén kötelező megadni minden futásnál.
- `optional`: Lista az opcionális paraméterekről ugyanezekkel a mezőkkel. A `flag` típusokat boolean kapcsolóként kell kezelni; az `option` típus értéket vár; a `config_option` olyan Hydra-szerű kulcs, amelyet `kulcs=érték` formában adnak át.

## Használati javaslatok
- Ha új szkriptet veszel fel, ügyelj arra, hogy a `script` útvonala illeszkedjen a tényleges fájlnévhez, és hogy a `required` / `optional` listák pontosan tükrözzék az `argparse` definíciókat.
- A `default` érték legyen JSON kompatibilis (szám, string, boolean vagy `null`). A CLI ebből tölti elő a mezőket.
- Ha a szkript környezeti változókat vagy titkos kulcsot igényel, az `api` mező segít a felületnek jelezni, hová kell a kulcsot tárolni (`keyholder.json`, env var stb.).
