# diarize_changes.py

GPU-ra optimalizált, offline beszélőváltás-detektor hosszú WAV fájlokra.

## Mire van kihegyezve
- filmhangból kivett beszédsáv
- sok beszélő
- a beszélők pontos elkülönítése helyett a váltási pontok és az overlap-határok
- akár 2 órás bemenet
- JSON kimenet

## Kimenet
```json
[
  {"start": 12.340, "end": 18.120, "type": "single"},
  {"start": 18.120, "end": 18.740, "type": "overlap"},
  {"start": 18.740, "end": 26.510, "type": "single"}
]
```

## Telepítés
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

A Hugging Face oldalon fogadd el a hozzáférést ezekhez:
- `pyannote/speaker-diarization-community-1`
- `pyannote/segmentation-3.0`

Majd állítsd be a tokent:
```bash
export HF_TOKEN="..."
```

## Futtatás
Legpontosabb preset:
```bash
python diarize_changes.py input.wav output.json --preset ultra --device cuda --max-speakers 50
```

Csak váltási pontok külön fájlba is:
```bash
python diarize_changes.py input.wav output.json   --output-change-points-json changes.json   --preset ultra --device cuda --max-speakers 50
```

Értékelés ground truth JSON ellen 100 ms toleranciával:
```bash
python diarize_changes.py input.wav output.json   --preset ultra --device cuda   --evaluate ground_truth.json --eval-tol-ms 100
```

## Presetek
- `fast`: gyorsabb, gyengébb finomítás
- `balanced`: általános használat
- `ultra`: lassabb, agresszívebb boundary-refinement

## Fontos megjegyzés
Ez a script a váltáspont-visszahívásra van optimalizálva. A valós, hosszú, sokbeszélős audión a ±100 ms pontosság nagyon jó esetben elérhető, de nem garantálható minden fájlon. A `--evaluate` módot érdemes használni a saját adataidon.
