# AI_Dubbing Project Structure

```
AI_Dubbing/
├── .gitignore
├── =12.3
├── access.log
├── config.json
├── infer_batch.log
├── keyholder.json
├── main_app.py
├── main_bash.py
├── README.md
├── requirements.txt
├── tree.MD
├── users.json
├── viewer.py
├── temp/
│   ├── c3e401aea35259abb46c17112cdd8832.mp4
│   └── e14448fe252ba306d0ac36c70268a122.mp4
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
│       └── README.md
├── scripts/
│   ├── ai_studio_code.html
│   ├── extract_audio_easy_channels.py
│   ├── f5_tts_easy_gai.py
│   ├── f5_tts_easy_glm.py
│   ├── f5_tts_easy.py
│   ├── fonetic.py
│   ├── merge_chunks_with_background_easy_multichannel.py
│   ├── merge_chunks_with_background_easy.py
│   ├── merge_to_video_easy_multichannal.py
│   ├── merge_to_video_easy.py
│   ├── normalise_and_cut_json_easy2.py
│   ├── separate_audio_easy_channels.py
│   ├── split_segments_by_speaker.py
│   ├── split_segments_by_speaker2.py
│   ├── split_segments_by_speaker3.py
│   ├── translate_chatgpt_srt_easy_glossary.py
│   ├── translate_chatgpt_srt_easy_tartalomigazítás.py
│   ├── translate_chatgpt_srt_easy_tartalomigazítás2.py
│   ├── translate_chatgpt_srt_easy.py
│   ├── translate_chatgpt5_srt_easy.py
│   ├── translate_chatgpt5.1_srt_easy.py
│   ├── translate.py
│   ├── unpack_srt_from_mkv_easy.py
│   ├── whisx_turbo.py
│   ├── whisx.py
│   ├── Nvidia_asr_eng/
│   │   ├── e2e_diarize_speech.py
│   │   ├── parakeet_transcribe_wordts_3.7.py
│   │   ├── parakeet_transcribe_wordts_3.8.py
│   │   ├── parakeet_transcribe_wordts_3.9.py
│   │   ├── parakeet_transcribe_wordts_4.0_easy.py
│   │   ├── parakeet_transcribe_wordts_4.0.py
│   │   ├── parakeet_transcribe_wordts_4.1_kisérleti.py
│   │   └── readme.MD
│   ├── old/
│   │   ├── 4, normalise_and_cut.py
│   │   ├── compare_srt.py
│   │   ├── DEEPSEEK_json_corrector_v2.py
│   │   ├── DEEPSEEK_json_corrector_v3.1.py
│   │   ├── DEEPSEEK_json_corrector_v3.2.py
│   │   ├── DEEPSEEK_json_corrector_v3.3.py
│   │   ├── DEEPSEEK_json_corrector_v3.4.py
│   │   ├── DEEPSEEK_json_corrector_v3.py
│   │   ├── DEEPSEEK_json_corrector.py
│   │   ├── diarize.py
│   │   ├── diarize2.py
│   │   ├── extract_audio_easy.py
│   │   ├── extract_audio.py
│   │   ├── f5_tts_easy.py
│   │   ├── f5_tts_easy2.py
│   │   ├── f5_tts_easy3.py
│   │   ├── f5_tts_easy4.py
│   │   ├── f5_tts_easy5.py
│   │   ├── f5_tts_easy6_kisérleti.py
│   │   ├── f5_tts_easy7_kisérleti.py
│   │   ├── f5_tts_easy8_kisérleti.py
│   │   ├── f5_tts_easy9_kisérleti_old.py
│   │   ├── f5_tts_easy9_kisérleti_old2.py
│   │   ├── f5_tts_easy9_kisérleti.py
│   │   ├── f5_tts_infer_API_config_01.py
│   │   ├── f5_tts_infer_API_config.py
│   │   ├── f5_tts_infer_API_model_config.py
│   │   ├── f5_tts_infer_API.py
│   │   ├── main_bash_2.py
│   │   ├── main_bash_3.py
│   │   ├── main_bash_4.py
│   │   ├── main_bash.py
│   │   ├── merge_chunks_with_background.py
│   │   ├── merge_to_video.py
│   │   ├── normalise_and_cut_json_easy.py
│   │   ├── normalise_and_cut_json.py
│   │   ├── normalise_and_cut.py
│   │   ├── OLLAMA_json_corrector_qwen.py
│   │   ├── OLLAMA_json_corrector_v2.py
│   │   ├── separate_audio_easy.py
│   │   ├── separate_audio.py
│   │   ├── splitter.py
│   │   ├── subtitle_extractor.py
│   │   ├── translate_chatgpt_srt.py
│   │   ├── translate_chatgpt.py
│   │   ├── unpack_srt_from_mkv.py
│   │   └── VAD.py
│   ├── OLD2/
│   │   ├── extract_audio_easy_channels_old.py
│   │   ├── f5_tts_easy9_kisérleti_old2.py
│   │   ├── f5_tts_easy9_kisérleti_old3.py
│   │   ├── f5_tts_easy9_kisérleti_old4.py
│   │   ├── fonetic_old.py
│   │   └── separate_audio_easy_channels_old.py
├── static/
│   ├── favicon.ico
│   └── css/
│       └── style.css
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── project.html
│   ├── review.html
│   └── viewer.html
├── tools/
├── transcripts/
├── TTS/
│   ├── e2_tts_base_config.json
│   ├── f5_tts_legacy_config.json
│   ├── f5_tts_v1_base_config.json
│   ├── f5_tts_v1_small_config.json
│   ├── TTS.md
│   ├── hun_v1_v1/
│   ├── hun_v1_v2.1/
│   ├── hun_v1_v2.2/
│   ├── hun_v1_v2.3/
│   ├── hun_v1_v3/
│   └── hun_v5/
└── workdir/
    └── [PROJECT_SUBDIRS]/
        ├── 1_upload/
        ├── 1.5_extracted_audio/
        ├── 2_separated_audio_background/
        ├── 2_separated_audio_speech/
        ├── 4_translated_json/
        ├── 5_translated_splits/
        ├── 6_noice_splits/
        ├── 7_film_dubbing/
        ├── 8_Download/
        └── logs/
