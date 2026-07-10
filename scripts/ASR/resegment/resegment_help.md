# resegment

**Futtatási környezet:** `sync`  
**Belépési pont:** `ASR/resegment/resegment.py`

## Mit csinál?
JSON szegmensek újraformázása - ASR script által létrehozott JSON fájlok biztonsági mentése és újraformázása különböző paraméterekkel.

A script a projekt ASR JSON fájljait újraszegmentálja. A szó-időbélyegek alapján új `segments` mezőt épít, opcionálisan energiaalapú időkorrekciót futtat, és opcionálisan pyannote-alapú extra speaker-váltási pontokat is figyelembe vesz.

## Kötelező paraméterek
- `project_name` (opció;  kapcsoló: `-p`, `--project-name`; alapértelmezés: nincs): A feldolgozandó projekt neve a `workdir` alatt.

## Opcionális paraméterek
- `max_pause` (opció;  kapcsoló: `--max-pause`; alapértelmezés: `0.3`): Legfeljebb ekkora szünetet hagy a script egy szegmensen belül. Nagyobb érték hosszabb mondatrészeket eredményezhet.
- `timestamp_padding` (opció;  kapcsoló: `--timestamp-padding`; alapértelmezés: `0.1`): Ennyi időt ad hozzá a szegmensek elejéhez és végéhez a kényelmesebb vágás érdekében.
- `max_segment_duration` (opció;  kapcsoló: `--max-segment-duration`; alapértelmezés: `11.5`): A szegmensek maximális hossza másodpercben.
- `enforce_single_speaker` (kapcsoló;  kapcsoló: `--enforce-single-speaker`; alapértelmezés: `false`): Arra kényszeríti az újraszegmentálást, hogy a meglévő szó-szintű speaker mezők váltásánál is új szegmens jöjjön létre. Alapállapotban ki van kapcsolva.
- `backup_disable` (kapcsoló;  kapcsoló: `--backup_disable`; alapértelmezés: `false`): Kikapcsolja a biztonsági mentés készítését. Alapállapotban a script mentést készít a módosított fájlokról; a kapcsoló csak ezt tiltja le.
- `skip_energy_refine` (kapcsoló;  kapcsoló: `--skip-energy-refine`; alapértelmezés: `false`): Kihagyja az energiaszint alapú finomítást, ezért gyorsabb, de kevésbé precíz lehet az illesztés. Alapállapotban ki van kapcsolva.
- `word_by_word_segments` (kapcsoló;  kapcsoló: `--word-by-word-segments`; alapértelmezés: `false`): A kimenetet a lehető legkisebb, szóközeli egységekre bontja. Alapállapotban ki van kapcsolva.
- `detect_speaker_changes` (kapcsoló;  kapcsoló: `--detect-speaker-changes`; alapértelmezés: `false`): Pyannote-alapú extra speaker-váltási pontokat detektál a hangfájlból, és ezeknél önálló szegmenst hoz létre. Ha `enforce_single_speaker` is be van kapcsolva, akkor a két logika összeadódik.
- `hf_token` (opció;  kapcsoló: `--hf-token`; alapértelmezés: nincs): Hugging Face token a pyannote modellek betöltéséhez. Ha a `detect_speaker_changes` be van kapcsolva, ez kötelező. A `HF_TOKEN` környezeti változóból is érkezhet.
- `speaker_change_preset` (opció;  kapcsoló: `--speaker-change-preset`; alapértelmezés: `balanced`): A speaker-váltás detektálás sebesség/pontosság presetje. Értékei: `fast`, `balanced`, `ultra`.
- `speaker_change_device` (opció;  kapcsoló: `--speaker-change-device`; alapértelmezés: `cuda`): A speaker-váltás detektálás futtatási eszköze. Értékei: `cuda`, `cpu`.
- `speaker_change_max_speakers` (opció;  kapcsoló: `--speaker-change-max-speakers`; alapértelmezés: `50`): A pyannote diarizáció felső becslése a beszélők számára.
- `speaker_change_min_region` (opció;  kapcsoló: `--speaker-change-min-region`; alapértelmezés: nincs): Felülírja a választott preset minimális régióhosszát másodpercben.
- `speaker_change_no_refine` (kapcsoló;  kapcsoló: `--speaker-change-no-refine`; alapértelmezés: `false`): Kikapcsolja a segmentation-3.0 alapú boundary refine lépést a speaker-váltás detektálásban.
- `debug` (kapcsoló;  kapcsoló: `--debug`; alapértelmezés: `false`): Részletes naplózást kapcsol be hibakereséshez. Alapállapotban ki van kapcsolva.

## Működési megjegyzések
- A script az eredeti JSON fájlokat felülírja, ezért érdemes csak indokolt esetben használni a `backup_disable` kapcsolót.
- A `detect_speaker_changes` opcióhoz a JSON mellé tartozó hangfájlra, működő pyannote környezetre és kötelező Hugging Face tokenre van szükség.
- A detektált plusz váltási pontok a legközelebbi szóhatárhoz igazodnak, és ott új szegmenst nyitnak.
- A kimeneti JSON `speaker_change_detection` és `speaker_change_points` mezőket is kaphat, ha az extra detektálás aktív.

## Hugging Face hozzáférés
A `detect_speaker_changes` opció a pyannote modelleket használja, ezért a Hugging Face oldalon előbb hozzáférést kell kérni és el kell fogadni a modellek feltételeit.

Engedélyezendő modellek:
- `pyannote/speaker-diarization-community-1`
- `pyannote/segmentation-3.0`

Javasolt lépések:
1. Lépj be a Hugging Face fiókodba.
2. Nyisd meg külön a két modell oldalát.
3. Kattints a hozzáféréskérés vagy licencelfogadás gombra, és fogadd el a feltételeket.
4. A Hugging Face beállításaidnál hozz létre egy read token-t.
5. Add meg futtatáskor `--hf-token` kapcsolóval, vagy állítsd be környezeti változóként:

```bash
export HF_TOKEN="..."
```

Ha a hozzáférés nincs elfogadva, a `--detect-speaker-changes` mód a token megléte mellett is hibával megállhat a pyannote modellek betöltésekor.

## Megjegyzés
A felületen a kapcsolók az alapértelmezett működési állapotot mutatják. Ha egy opció negatív CLI kapcsolóval működik, a webes jelölő ettől függetlenül a tényleges funkció állapotát jelzi.
