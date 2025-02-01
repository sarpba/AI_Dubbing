# AI Dubbing Project Structure

```bash
AI_Dubbing/
├── normalisers/
│   ├── hun/
│   │   ├── changes.csv          # Custom Hungarian replacement rules
│   │   ├── force_changes.csv    # Mandatory text substitutions
│   │   └── normaliser.py        # Hungarian-specific normalization logic
│   └── simple_normaliser_for_any_language/
│       ├── changes.csv          # Generic replacement rules
│       ├── force_changes.csv    # Basic mandatory substitutions
│       └── normaliser.py        # Language-agnostic normalization base
├── scripts/
│   ├── audio_cuter_normaliser.py  # Audio cutting and normalization
│   ├── audio_separate.py         # Voice/sound separation
│   ├── copy_numbers_and_spec.py  # Special character handling
│   ├── f5_tts_infer_API.py       # TTS API integration
│   ├── merge_audio.py            # Audio file merging
│   ├── merge_video.py            # Video file merging
│   ├── splitter.py               # File splitting utilities
│   ├── translate.py              # Translation core logic
│   └── whisx.py                  # Whisper model integration
├── tabs/
│   ├── adjust_audio.py           # Audio adjustment UI
│   ├── audio_splitting.py        # Audio segmentation interface
│   ├── compare_transcripts.py    # Transcript comparison tools
│   ├── integrate_audio.py        # Audio integration logic
│   ├── project_creation.py       # Project setup utilities
│   ├── transcription.py          # Main transcription interface
│   ├── translate.py              # Translation interface
│   ├── tts_generation.py         # TTS synthesis UI
│   └── verify_chunks.py          # Validation utilities
├── tools/                        # Additional utilities
├── main_app.py                   # Main application entry point
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Key File Purposes:
- `normalisers/`: Language-specific text normalization rules
- `scripts/`: Core audio/video processing pipelines
- `tabs/`: GUI components for different workflow stages
- `*.csv` files: Configuration for text replacements and transformations
