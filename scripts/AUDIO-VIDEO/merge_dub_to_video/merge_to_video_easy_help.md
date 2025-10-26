# merge_to_video_easy – konfigurációs útmutató

**Futtatási környezet:** `sync`  
**Belépési pont:** `merge_to_video_easy.py`

A szkript a kész szinkronhangot és – ha rendelkezésre áll – a feliratot összemásolja az eredeti videóval. Az eredményt a projekt `deliveries` (vagy konfigurációban megadott) mappájába helyezi.

## Kötelező beállítások
- `project_name` (pozícionális, alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.
- `language` (`-lang`, `--language`, option, alapértelmezés: nincs): A muxolás során a hozzáadott audió nyelvi címkéje (pl. `hun`, `eng`). A konténer metaadataiban jelenik meg.

## Opcionális beállítás
- `debug` (`--debug`, flag, alapértelmezés: `false`): Extra naplózás a futásról.
