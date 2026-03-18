# SAM_audio

**Futtatási környezet:** `sam-audio`  
**Belépési pont:** `AUDIO-VIDEO/SAM_audio/SAM_audio.py`

## Mit csinál?
SAM-Audio ONNX wrapper a beszéd és háttér szétválasztására az extracted_audio bemenetből.

A script a projekt audio- és videófájljait készíti elő, alakítja át vagy fűzi össze a szinkronizálási pipeline következő lépéseihez.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `prompt` (opció;  kapcsoló: `--prompt`; alapértelmezés: `speech`): A szeparáló modellnek adott rövid utasítás. Meghatározza, hogy milyen hangkomponenst emeljen ki.
- `device` (opció;  kapcsoló: `--device`; alapértelmezés: `cuda`): A futtatás eszköze, például `cpu`, `cuda`, `cuda:0` vagy `mps`.
- `onnx_script` (opció;  kapcsoló: `--onnx-script`; alapértelmezés: `onnx_inference.py`): Az ONNX futtatást végző segédszkript neve vagy útvonala.
- `onnx_model_dir` (opció;  kapcsoló: `--onnx-model-dir`; alapértelmezés: `onnx_models`): Az ONNX modellfájlokat tartalmazó könyvtár.
- `onnx_steps` (opció;  kapcsoló: `--onnx-steps`; alapértelmezés: `16`): Az ONNX inferencia lépésszáma.
- `sample_rate` (opció;  kapcsoló: `--sample-rate`; alapértelmezés: `48000`): A feldolgozás cél mintavételi frekvenciája.
- `mono` (kapcsoló;  kapcsoló: `--mono`; alapértelmezés: `false`): A bemenetet vagy a köztes fájlokat monóvá alakítja. Alapállapotban ki van kapcsolva.
- `force_convert` (kapcsoló;  kapcsoló: `--force-convert`; alapértelmezés: `false`): A script akkor is újrakonvertálja a bemenetet, ha használható formátumban már létezik. Alapállapotban ki van kapcsolva.
- `keep_temp` (kapcsoló;  kapcsoló: `--keep-temp`; alapértelmezés: `false`): Megtartja az ideiglenes fájlokat a futás után is. Alapállapotban ki van kapcsolva.
- `predict_spans` (kapcsoló;  kapcsoló: `--predict-spans`; alapértelmezés: `false`): Bekapcsolja az időtartamok vagy aktivitási szakaszok becslését. Alapállapotban ki van kapcsolva.
- `reranking_candidates` (opció;  kapcsoló: `--reranking-candidates`; alapértelmezés: `1`): Ennyi jelöltből választja ki az utólagos rangsorolás a legjobbat.
- `span_threshold` (opció;  kapcsoló: `--span-threshold`; alapértelmezés: `0.3`): A span előrejelzés küszöbértéke.
- `chunk_seconds` (opció;  kapcsoló: `--chunk-seconds`; alapértelmezés: `60`): A hosszú fájlok feldolgozási darabjainak mérete másodpercben.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
