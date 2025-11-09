# AI Dubbing

<p align="center">
  <a href="https://youtu.be/r5vSj6TRk8Y" target="_blank" rel="noopener noreferrer">
    <img src="static/logo.png" alt="AI Dubbing logo" width="240">
  </a>
</p>

## Overview
AI Dubbing is a toolkit for multilingual dubbing of videos and audio. The pipeline combines WhisperX-based transcription, optional speaker diarization, and multiple TTS models to produce the final track.
The `main_app.py` Flask front-end reads `scripts/scripts.json` to expose every module under `scripts/`, so the whole workflow can be orchestrated from a single UI.

## Installation and Environment Setup
1. Install the base `sync` conda environment by following the [sync-linux setup guide](ENVIROMENTS/sync-linux.md); this is the primary workflow environment.
2. Provision the specialized environments as needed (every guide targets Ubuntu 24.04):
   - [whisperx-linux setup guide](ENVIROMENTS/whisperx-linux.md) – for multi-GPU WhisperX ASR and alignment.
   - [nemo-linux setup guide](ENVIROMENTS/nemo-linux.md) – for NVIDIA NeMo-based Canary/Parakeet ASR and diarization.
   - [f5-tts-linux setup guide](ENVIROMENTS/f5-tts-linux.md) – for F5-TTS models and the related normalization helpers.
   - [vibevoice-linux setup guide](ENVIROMENTS/vibevoice-linux.md) – for running the VibeVoice TTS pipeline.
   - [demucs-linux setup guide](ENVIROMENTS/demucs-linux.md) – for Demucs/MDX speech–background separation.
3. Always launch `main_app.py` from the `sync` environment; the remaining specialized environments are activated automatically by the application when needed.

## Model and API Preparation
- Update `/anaconda3/envs/whisperx/lib/python3.10/site-packages/whisperx/alignment.py` to point to more accurate default alignment models; this improves timeline precision during transcription.
- For F5-TTS, copy your `model.pt`, `vocab.txt`, and `model_conf.json` into the appropriate `TTS/XXX` subdirectory. Configuration templates are available in the `TTS` folder.
- For VibeVoice, change into the `TTS` directory and clone the desired Hugging Face repository (for example: `git clone https://huggingface.co/sarpba/VibeVoice-large-HUN`).
- (Optional) Create a Hugging Face account, accept the Pyannote Speaker Diarization 3.1 license, and securely store a read-only API token: https://huggingface.co/pyannote/speaker-diarization-3.1
- Register for a DeepL account, enable the free API tier, and generate an API key (500,000 characters/month, roughly 10–20 hours of video).
- Create an ElevenLabs account to access the ASR module; the free tier provides roughly 2–3 hours of credits per month.

## Pipeline Modules and Script Catalog

The summary below is based on the `scripts/` folder’s `*_help.md` files. Each entry highlights the recommended conda environment; consult the referenced help documents for the full CLI surface.

### Input Preparation
- `scripts/AUDIO-VIDEO/audio_copy_helper/audio-copy-helper.py` (`sync`) – Converts uploaded audio to 44.1 kHz WAV and copies it into the requested project subfolders (`--extracted_audio`, `--separated_audio_*`); requires `-p/--project-name`.
- `scripts/AUDIO-VIDEO/extract_audio_from_video/extract_audio_easy_channels.py` (`sync`) – Pulls the first video from `upload`, extracts its audio track, optionally splits each channel (`--keep_channels`), and stores the result in `extracted_audio`.
- `scripts/AUDIO-VIDEO/separate_speak_and_background_audio/separate_audio_easy.py` (`demucs`) – Runs Demucs/MDX models to split speech and background audio; `--models`, `--chunk_size`, and `--background_blend` fine-tune the separation.
- `scripts/AUDIO-VIDEO/unpack_srt_from_mkv/unpack_srt_from_mkv_easy.py` (`sync`) – Extracts embedded `.srt` subtitle tracks from the uploaded MKV into the project’s `subtitles` directory.

### ASR and Resegmentation
- `scripts/ASR/canary-non_working_beta/canary-easy.py` (`nemo`) – Runs NVIDIA Canary ASR on `separated_audio_speech` with automatic or fixed chunks and optional translation (`--source-lang`, `--target-lang`, `--keep_alternatives`).
- `scripts/ASR/elevenlabs/elevenlabs.py` (`sync`) – Calls the ElevenLabs STT API, writes normalized `word_segments` JSON files beside every audio clip, and manages the API key through `keyholder.json` (`--api-key`, `--diarize`).
- `scripts/ASR/paraket-eng/parakeet-tdt-0.6b-v2.py` (`nemo`) – Uses the Parakeet TDT 0.6B v2 model with automatic chunk calibration; segment boundaries can be tuned via `--max-pause`, `--timestamp-padding`, and `--max-segment-duration`.
- `scripts/ASR/paraket-multilang/parakeet-tdt-0.6b-v3.py` (`nemo`) – Multilingual Parakeet TDT v3 pipeline that reuses the same CLI but covers more languages out of the box.
- `scripts/ASR/whisperx/whisx_v1.1.py` (`whisperx`) – Runs WhisperX `large-v3` across multiple GPUs, optionally adds pyannote diarization (`--hf_token`, `--gpus`, `--timestamp-padding`), and stores the output alongside `separated_audio_speech`.
- `scripts/ASR/resegment/resegment.py` (`sync`) – Resections ASR JSON with energy-based boundary correction and optional backups (`--max-pause`, `--timestamp-padding`, `--enforce-single-speaker`).
- `scripts/ASR/resegment-mfa/resegment-mfa.py` (`sync`) – Integrates Montreal Forced Aligner to refine word timestamps (`--use-mfa-refine`) while exposing the standard resegment switches.

### Diarization
- `scripts/DIARIZE/speaker_diarize_e2e_non_working_beta/e2e_diarize_speech.py` (`nemo`) – Sortformer-based end-to-end diarization with Hydra configuration and optional Optuna post-processing search.
- `scripts/DIARIZE/speaker_diarize_pyannote/split_segments_by_speaker.py` (`sync`) – Applies pyannote diarization to existing segments (`--hf-token`, `--min-chunk`, `--no-backup`, `--dry-run`) and writes the updated JSON in place or via suffix.

### Translation
- `scripts/TRANSLATE/chatgpt_with_srt/translate_chatgpt_srt_easy.py` (`sync`) – Uses the ChatGPT API to translate segments, optionally injecting SRT context, while storing the API key in `keyholder.json` (`-input_language`, `-output_language`, `-model`, `-auth_key`).
- `scripts/TRANSLATE/deepl/deepl_translate.py` (`sync`) – Calls the DeepL REST API and writes translated JSON files into `translated`; caches the key in `keyholder.json` (`--auth-key`) and falls back to `config.json` for default languages.

### TTS
- `scripts/TTS/f5-tts/f5_tts_easy_EQ.py` (`f5-tts`) – Core F5-TTS pipeline with optional EQ shaping and Whisper verification (`--norm`, `--eq-config`, `--max-retries`, `--whisper-model`).
- `scripts/TTS/f5-tts-narrator/f5_tts_narrator.py` (`f5-tts`) – Uses an external narrator directory (`--narrator`) while retaining F5-TTS multi-pass verification.
- `scripts/TTS/vibevoice_old/vibevoice-tts.py` (`vibevoice`) – Baseline VibeVoice TTS with LoRA support, optional EQ, retry logic, and Whisper-based QA.
- `scripts/TTS/vibevoice_1.1/vibevoice-tts_v1.1.py` (`vibevoice`) – Assigns workers to GPUs deterministically and saves reference snippets for each failed attempt to ease debugging.
- `scripts/TTS/vibevoice_1.2/vibevoice-tts_v1.2.py` (`vibevoice`) – Adds pitch-based validation and retry control (`--enable_pitch_check`, `--pitch_tolerance`, `--pitch_retry`) on top of the earlier improvements.
- `scripts/TTS/vibevoice-narrator/vibevoice-tts_narrator.py` (`vibevoice`) – Relies on a single narrator WAV (`--narrator`) while preserving the VibeVoice EQ/normalization workflow and Whisper checks.

### Audio Mastering and Delivery
- `scripts/AUDIO-VIDEO/normalize_and_cut_tts_files_old_version/normalise_and_cut_json_easy.py` (`sync`) – Matches segment loudness to the reference via RMS, trims silence with VAD, and optionally deletes empty files (`--delete_empty`, `--no-sync-loudness`).
- `scripts/AUDIO-VIDEO/normalize_and_cut_tts_files/normalise_and_cut_json_easy_v1.1.py` (`sync`) – Uses FFmpeg `silenceremove`, peak limiting, and the same normalization flags for cleaner segment tails.
- `scripts/AUDIO-VIDEO/merge_chunks_with_background/merge_chunks_with_background_easy.py` (`sync`) – Concatenates `translated_splits`, optionally mixes background audio in narrator mode (`--background-volume`), and writes the result to `film_dubbing`.
- `scripts/AUDIO-VIDEO/merge_chunks_with_bacground_v1.1/merge_chunks_with_background_easy.py` (`sync`) – Extends the previous script with `--time-stretching` to shrink overlaps without altering pitch.
- `scripts/AUDIO-VIDEO/merge_chunks_with_background_narrator_version/merge_chunks_with_background_beta.py` (`sync`) – Narrator-focused beta variant that controls overlap compression via `--max-speedup` while mixing the background track.
- `scripts/AUDIO-VIDEO/merge_dub_to_video/merge_to_video_easy.py` (`sync`) – Muxes the final dub and optional subtitles back into the source video, tagging the audio stream with `--language`, and publishes the output under `deliveries`.

### Quality and Helper Tools
- `scripts/AUDIO-VIDEO/pitch_analizer/pitch_analizer.py` (`sync`) – Compares segment pitch against a narrator reference (`--tolerance`, `--delete-outside`) and can automatically remove outliers.
- `scripts/NORMALIZER_HELPER/collect_normalized_translations.py` (`f5-tts`) – Aggregates normalized `translated` texts for the Hungarian normalizer, targeting a project supplied through `-p/--project`.
- `scripts/TTS/EQ.json` – Multi-band EQ definition consumed by the F5-TTS and VibeVoice scripts via `--eq-config`; tune `global_gain_db` and `points` to match your reference audio.

### Developer Documentation
- `scripts/How_to_create_new_script_modul/how_to_create_scripts_help.md` – Developer guide for config-driven scripts: locating `config.json`, exposing `-p/--project-name`, enabling debug flags, and handling errors consistently.
- `scripts/How_to_create_new_script_modul/how_to_create_scripts_json_help.md` – Documents the `scripts.json` schema (`enviroment`, `required`, `optional`) so CLI definitions stay synchronized with the UI.

## Running the Application
```bash
conda activate sync
cd AI_Dubbing
python main_app.py
```


## External links (if you use runpod, please use my template, this way you support my work with 1% of the money you spend)

Docker: sarpba/ai_dubbing:latest
Runpod: https://console.runpod.io/deploy?template=5kqz4o0l31&ref=jwyyc9tv
