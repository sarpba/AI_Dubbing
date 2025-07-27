# AI_Dubbing_new Project Structure

```
.
├── config.json
├── main_app.py
├── README.md
├── requirements.txt
├── normalisers/
│   ├── hun/
│   │   ├── changes.csv
│   │   ├── force_changes.csv
│   │   ├── normaliser.py
│   │   ├── normalizer_test.py
│   │   └── README.md
│   └── simple_normaliser_for_any_language/
│       ├── changes.csv
│       ├── force_changes.csv
│       ├── normaliser.py
├── scripts/
│   ├── 4, normalise_and_cut.py
│   ├── compare_srt.py
│   ├── diarize.py
│   ├── diarize2.py
│   ├── extract_audio.py
│   ├── f5_tts_infer_API.py
│   ├── merge_chunks_with_background.py
│   ├── merge_to_video.py
│   ├── normalise_and_cut_json.py
│   ├── normalise_and_cut.py
│   ├── separate_audio.py
│   ├── splitter.py
│   ├── subtitle_extractor.py
│   ├── translate.py
│   ├── VAD.py
│   ├── whisx_turbo.py
│   ├── whisx.py
│   └── Nvidia_asr_eng/
│       ├── parakeet_transcribe_wordts_3.7.py
│       └── readme.MD
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── index.html
│   ├── project.html
│   └── review.html
├── tools/
├── transcripts/
├── TTS/
└── workdir/
    └── [PROJECT_SUBDIRS]/
        ├── [upload]/
        ├── [extracted_audio]/
        ├── [separated_audio_background]/
        ├── [separated_audio_speech]/
        ├── [translated]/
        ├── [translated_splits]/
        ├── [noice_splits]/
        ├── [film_dubbing]/
        ├── [download]/
        └── [logs]/
