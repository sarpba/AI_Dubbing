# Resegment Script - JSON szegmensek újraformázása

## Áttekintés

Ez a script lehetővé teszi az ASR (Automatic Speech Recognition) scriptek által létrehozott JSON fájlok újraformázását különböző szegmentálási paraméterekkel. A script biztonsági mentést készít az eredeti JSON fájlokról, majd új JSON fájlokat hoz létre testreszabott szegmentálási beállításokkal.

## Főbb funkciók

- **Biztonsági mentés**: Az eredeti JSON fájlok biztonsági mentése `.json.bak` kiterjesztéssel
- **Energia-alapú korrekció**: FFT/RMS energiabecsléssel pontosított szó időbélyegek a szegmentálás előtt
- **Újraformázás**: Az eredeti JSON fájlok felülírása újraformázott tartalommal
- **Konfigurálható szegmentálás**: Testreszabható szegmentálási paraméterek
- **Projekt-alapú működés**: A config.json alapján automatikusan feloldja a projekt könyvtárait

## Használat

### Alapvető használat

```bash
python scripts/ASR/resegment/resegment.py -p <projekt_név>
```bash

### Példa paraméterekkel

```json
python scripts/ASR/resegment/resegment.py \
  -p MyProject \
  --max-pause 1.0 \
  --timestamp-padding 0.2 \
  --max-segment-duration 15.0 \
  --enforce-single-speaker \
  --no-backup
```

## Paraméterek

### Kötelező paraméterek

- `-p, --project-name`: A projekt neve (a workdir alatti mappa)

### Opcionális paraméterek

- `--max-pause`: Mondatszegmensek közti maximális szünet másodpercben (alapértelmezett: 0.8)
- `--timestamp-padding`: Szó időbélyegek bővítése másodpercben (alapértelmezett: 0.1)
- `--max-segment-duration`: Mondatszegmensek maximális hossza másodpercben (alapértelmezett: 11.5)
- `--enforce-single-speaker`: Speaker diarizáció alapján szegmentálás (alapértelmezett: ki)
- `--skip-energy-refine`: Energia-alapú (FFT/RMS) szó időbélyeg korrekció kihagyása
- `--backup`: JSON fájlok biztonsági mentése (alapértelmezett: be)
- `--no-backup`: JSON fájlok biztonsági mentésének kikapcsolása
- `--debug`: Debug mód engedélyezése

## Kimenet

A script a következő fájlokat hozza létre:

1. **Biztonsági mentés**: `<eredeti_fájl>.json.bak` (ha a backup engedélyezve van)
2. **Újraformázott JSON**: Az eredeti JSON fájl felülírása a resegmentált tartalommal

### Újraformázott JSON struktúra

```
{
  "segments": [...],
  "word_segments": [...],
  "language": "hu",
  "provider": "resegment",
  "diarization": false,
  "original_provider": "elevenlabs",
  "resegment_parameters": {
    "max_pause_s": 0.8,
    "padding_s": 0.1,
    "max_segment_s": 11.5,
    "enforce_single_speaker": false,
    "energy_refine": true,
    "energy_window_ms": 10
  },
  "energy_adjustment": {
    "status": "applied",
    "audio": "clip.wav",
    "frame_duration_ms": 10,
    "threshold_high": 820.5,
    "threshold_low": 410.2,
    "frames": 312
  }
}
```

## Energia-alapú pontosítás

A script megpróbálja a JSON-nal azonos nevű hangfájlt (pl. `clip.wav`, `clip.mp3`) betölteni, majd rövid ablakokra bontva FFT/RMS energiát számol. Ha egy szó elején azonnal erős energia látható, a feldolgozás addig lépked vissza 10 ms-es keretekben, amíg az első csendes ablakot meg nem találja; ha viszont a start körül csend van, előrefelé keresi meg az első energikus ablakot. A szó végén, ha a pillanatnyi végponton csend van, hasonlóan visszafelé lépked az első energiatüskéig, és annak környezetéhez igazítja a határt; amennyiben pedig még energia van a végponton, először +0,2 s tartományban keresi a rákövetkező csendet, majd ha ott sincs, legfeljebb −0,3 s-ig visszafelé vizsgál csendet. A talált csend kezdete lesz az új szóvég, és ha emiatt átfedne a következő szóval, annak időbélyegeit is hátrébb tolja a feldolgozás. A folyamat az `ffmpeg` eszközt használja szükség esetén konverzióra; ha az `ffmpeg` nem elérhető vagy a hangfájl hiányzik, a korrekció kimarad, és erről napló készül.

## Szegmentálási algoritmus

A script az elevenlabs.py-ból átvett szegmentálási algoritmust használja:

1. **Szó szegmensek kinyerése**: A JSON fájlból kinyeri a szó szintű időbélyegeket
2. **Időbélyeg bővítés**: A szó időbélyegeket bővíti a megadott padding értékkel
3. **Mondatszegmentálás**: Szünetek és/vagy speaker változások alapján szegmensekre bontja
4. **Hosszú szegmensek felosztása**: Túl hosszú szegmenseket felosztja mondatvégi jelek vagy egyenlő részek alapján

## Használati esetek

### 1. Finomhangolt szegmentálás
Ha az eredeti ASR túl rövid vagy túl hosszú szegmenseket hozott létre, a paraméterek módosításával optimalizálhatjuk a szegmentálást.

### 2. Speaker-alapú szegmentálás
Ha diarizált adatokkal rendelkezünk, az `--enforce-single-speaker` kapcsolóval biztosíthatjuk, hogy minden szegmens csak egy beszélőt tartalmazzon.

### 3. Biztonsági mentés és tesztelés
A script lehetővé teszi az eredeti fájlok biztonsági mentését, miközben különböző paraméterekkel teszteljük az új szegmentálást.

## Példa munkafolyamat

1. **Eredeti ASR futtatása**:
   ```bash
   python scripts/ASR/elevenlabs/elevenlabs.py -p MyProject
   ```

2. **Resegmentálás tesztelése**:
   ```bash
   python scripts/ASR/resegment/resegment.py -p MyProject --max-pause 1.0 --no-backup
   ```

3. **Végső resegmentálás**:
   ```bash
   python scripts/ASR/resegment/resegment.py -p MyProject --max-pause 0.6 --timestamp-padding 0.15
   ```

## Hibakezelés

- **Hiányzó projekt**: Ha a projekt mappa nem található, a script hibát jelez és leáll
- **Érvénytelen JSON**: Ha a JSON fájl sérült, a script kihagyja azt és folytatja a következővel
- **Mentési hiba**: Ha az új JSON fájl mentése sikertelen, a script naplózza a hibát és folytatja

## Megjegyzések

- A script csak `.json` kiterjesztésű fájlokat dolgoz fel
- Az eredeti fájlok felülírása előtt biztonsági mentést készít (ha engedélyezve van)
- A script megőrzi az eredeti JSON metaadatokat és hozzáadja a resegmentálási paramétereket
