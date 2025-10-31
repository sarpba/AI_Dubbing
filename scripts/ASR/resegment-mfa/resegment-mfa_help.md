# Resegment MFA Script – JSON szegmensek MFA alapú finomítása

## Áttekintés

Ez a script az ASR által készített JSON fájlokat szegmentálja újra, miközben opcionálisan a Montreal Forced Aligner (MFA) segítségével finomítja a szó időbélyegeket. A folyamat elsősorban angol nyelvű anyagokra van optimalizálva, és az MFA `english_mfa` akusztikus modelljét, valamint szótárát használja. A script biztonsági mentést készít az eredeti JSON fájlról (kikapcsolható), majd az újraszegmentált tartalmat ugyanabba a fájlba írja vissza.

## Főbb funkciók

- **MFA integráció**: Szó időbélyegek pontosítása a Montreal Forced Alignerrel
- **Rugalmas szegmentálás**: Állítható szünet, időbélyeg padding és szegmenshossz paraméterek
- **Speaker-alapú bontás**: Igény szerint szétszedi a szegmenseket beszélőváltásnál
- **Egyszavas mód**: Minden szó külön szegmensbe helyezhető
- **Biztonsági mentés**: Opcionális `.json.bak` mentés az eredeti állományról

## Előfeltételek

- Telepített `ffmpeg` eszköz a hangfájlok 16 kHz mono WAV formátumba konvertálásához
- Telepített Montreal Forced Aligner (`pip install montreal-forced-aligner`, vagy hivatalos binary)
- Letöltött MFA erőforrások:
  ```bash
  mfa model download acoustic english_mfa
  mfa model download dictionary english_mfa
  ```
- Python `textgrid` csomag:
  ```bash
  pip install textgrid
  ```

## Használat

### Alapvető futtatás

```bash
python scripts/ASR/resegment-mfa/resegment-mfa.py -p <projekt_név>
```

### Példa MFA finomítással

```bash
python scripts/ASR/resegment-mfa/resegment-mfa.py \
  -p MyEnglishProject \
  --max-pause 0.9 \
  --timestamp-padding 0.15 \
  --max-segment-duration 12.0 \
  --enforce-single-speaker \
  --use-mfa-refine
```

### Tippek

- A `--use-mfa-refine` kapcsoló csak akkor működik, ha az adott JSON-hoz tartozó hangfájl (azonos fájlnévvel és `.wav/.mp3/.flac/.m4a/.ogg` kiterjesztéssel) elérhető.
- Ha az MFA nem található vagy hibát ad, a script kihagyja az időbélyeg finomítást, és naplózza az okot.

## Paraméterek

### Kötelező

- `-p`, `--project-name`: A projekt neve (a `config.json` `DIRECTORIES.workdir`/`<projekt>` alatt)

### Opcionális

- `--max-pause`: Szavak közötti maximális szünet másodpercben (alapértelmezett: 0.8)
- `--timestamp-padding`: Szó időbélyegek bővítése másodpercben (alapértelmezett: 0.1)
- `--max-segment-duration`: Egy szegmens maximális hossza másodpercben (alapértelmezett: 11.5)
- `--enforce-single-speaker`: Szegmens bontása beszélőváltáskor (diarizáció szükséges)
- `--no-backup`: Biztonsági mentés kikapcsolása
- `--use-mfa-refine`: MFA-alapú szó időbélyeg finomítás bekapcsolása
- `--word-by-word-segments`: Minden szó külön szegmens
- `--debug`: Részletes naplózás bekapcsolása

## Kimenet

1. **Biztonsági mentés**: `<fájl>.json.bak` – ha a backup engedélyezve van
2. **Új JSON**: Az eredeti állomány felülírása az újraszegmentált tartalommal

### JSON példa (részlet)

```json
{
  "segments": [...],
  "word_segments": [...],
  "provider": "resegment_mfa",
  "resegment_parameters": {
    "max_pause_s": 0.8,
    "padding_s": 0.1,
    "max_segment_s": 11.5,
    "enforce_single_speaker": false,
    "mfa_refine": true
  },
  "alignment_adjustment": {
    "status": "applied_mfa_alignment",
    "audio": "clip.wav"
  }
}
```

## Hibakezelés

- **Hiányzó hangfájl**: Az MFA finomítás kimarad, a script figyelmeztetést ír a logba
- **Érvénytelen JSON**: A fájl feldolgozása sikertelennek jelölődik, a script a következővel folytatja
- **Mentési probléma**: A mentés hibáját a log tartalmazza, az eredeti fájl változatlan marad

## Ajánlott munkafolyamat

1. **ASR futtatása**: pl. `python scripts/ASR/elevenlabs/elevenlabs.py -p MyEnglishProject`
2. **Resegment MFA teszt**: `python scripts/ASR/resegment-mfa/resegment-mfa.py -p MyEnglishProject --use-mfa-refine --no-backup`
3. **Finomhangolás**: Paraméterek igazítása (`--max-pause`, `--timestamp-padding`) és végleges futtatás backup-pal

## Megjegyzések

- A script csak `.json` kiterjesztésű állományokat dolgoz fel az adott projekt `separated_audio_speech` mappájában.
- Az MFA futása időigényes lehet; nagy projektek esetén célszerű előbb kisebb mintán tesztelni.
- Ha a `textgrid` modul nincs telepítve, a script nem indul el – telepítése kötelező MFA módhoz.
