# File Structure

This document describes the directory structure of the AI_Dubbing project.

```
/home/sarpba/AI_Dubbing
├── .gitignore
├── config.json
├── main_app.py
├── README.md
├── normalisers
│   ├── hun
│   │   ├── changes.csv
│   │   ├── force_changes.csv
│   │   └── normaliser.py
│   └── simple_normaliser_for_any_language
│       ├── changes.csv
│       ├── force_changes.csv
│       └── normaliser.py
├── scripts
│   ├── f5_tts_infer_API.py
│   ├── merge_chunks_with_background.py
│   ├── merge_to_video.py
│   ├── normalise_and_cut.py
│   ├── separate_audio.py
│   ├── splitter.py
│   ├── translate.py
│   ├── whisx_turbo.py
│   └── whisx.py
├── TTS
└── workdir
