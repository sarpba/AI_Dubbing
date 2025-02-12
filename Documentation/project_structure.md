# Project Directory Structure

The AI_ Dubbing project follows a modular organization with distinct directories for different types of functionality. Here' is the detailed directory structure with additional comments:

AI_ Dubbing/
├── .gitignore                          # Configuration file for Git to ignore specific files/directories
├── README.md                           # Project documentation and setup instructions
├── config.json                         # Main configuration file containing project settings
├── main_app.py                         # Core application entry point
├── requirements.txt                    # List of Python dependencies
├── normalisers/                        # Directory for text normalization logic
│   ├── hun/                            # Hungarian language specific normalization
│   │   ├── changes.csv                 # Custom replacement rules for Hungarian
│   │   ├── force_ changes.csv          # Mandatory substitutions that must always be applied
│   │   └── normaliser.py               # Implements Hungarian-specific normalization logic
│   └── simple_ normaliser_for_any_language/
│       ├── changes.csv                # General language replacement rules
│       ├── force_ changes.csv          # Mandatory substitutions for general use
│       └── normaliser.py               # Generic text normalization implementation
├── scripts/                            # Collection of utility scripts
│   ├── compare_srt.py                  # Compares subtitle files
│   ├── f5_tts_infer_API.py             # Text-to-speech inference using F5 API
│   ├── merge_chunks_with_background.py # Merges audio chunks with background music
│   ├── merge_to_video.py               # Combines audio and subtitles into final video
│   ├── normalise_and_cut.py            # Normalizes text and handles segmentation
│   ├── separate_audio.py               # Separates audio streams from video files
│   ├── splitter.py                     # Splits media files into chunks
│   ├── subtitle_extractor.py           # Extracts subtitles from media files
│   ├── translate.py                    # Translates text content
│   └── whisx_turbo.py                  # Optimized version of the WhisperX model for faster processing
├── tools/                              # External tools and dependencies
├── TTS/                                # Text-to-speech modells is subdirectorys
└── workdir/                            # Working directory for temporary files and intermediate results


This structure organizes the project into distinct functional areas:
- `normalisers/`: Contains language-specific normalization logic with separate directories for Hungarian (`hun`) and a general-purpose normaliser.
- `scripts/`: Holds various utility scripts for different aspects of the AI dubbing process, including subtitle handling, audio processing, and translation.
- `tools/`: Contains external tools and configurations, including TTS (Text-to-Speech) capabilities.
- `workdir/`: Used for temporary files and intermediate results during processing.

The root directory contains essential project files like `config.json` and `requirements.txt`, as well as core application files like `main_app.py`.
