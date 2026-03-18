# AI Dubbing

<p align="center">
  <a href="https://youtu.be/r5vSj6TRk8Y" target="_blank" rel="noopener noreferrer">
    <img src="static/logo.png" alt="Logo de AI Dubbing" width="240">
  </a>
</p>

## Descripción general (ES)
AI Dubbing es un conjunto de herramientas para doblaje multilingüe de video y audio. El flujo combina transcripción basada en WhisperX, separación opcional de hablantes y varios modelos TTS para producir la salida final.
`main_app.py` proporciona una interfaz de control basada en Flask que ofrece los pasos ejecutables a partir de un catálogo de scripts generado automáticamente desde `scripts/`, de modo que todo el workflow puede iniciarse desde un solo lugar.

## Guía de instalación y entornos (ES)
1. Instala el entorno base de conda `sync` siguiendo la [guía de instalación sync-linux](ENVIROMENTS/sync-linux.md); es el entorno principal del workflow.
2. Para los componentes especializados del pipeline, crea entornos separados; todas las guías están preparadas para Ubuntu 24.04:
   - [guía de instalación whisperx-linux](ENVIROMENTS/whisperx-linux.md): para ejecutar WhisperX ASR y aligner con varias GPU.
   - [guía de instalación nemo-linux](ENVIROMENTS/nemo-linux.md): para ASR y diarización basados en NVIDIA NeMo Canary/Parakeet.
   - [guía de instalación f5-tts-linux](ENVIROMENTS/f5-tts-linux.md): para modelos F5-TTS y scripts auxiliares de normalización.
   - [guía de instalación vibevoice-linux](ENVIROMENTS/vibevoice-linux.md): para ejecutar el pipeline TTS de VibeVoice.
   - [guía de instalación demucs-linux](ENVIROMENTS/demucs-linux.md): para separación de voz y fondo basada en Demucs/MDX.
3. Ejecuta siempre `main_app.py` desde el entorno `sync`; el sistema gestiona automáticamente los demás entornos especializados durante la ejecución.

## Preparación de modelos y API (ES)
- Actualiza el archivo `/anaconda3/envs/whisperx/lib/python3.10/site-packages/whisperx/alignment.py` y establece modelos de alineación por defecto más precisos para mejorar la exactitud temporal.
- Para usar F5-TTS, copia tus archivos `model.pt`, `vocab.txt` y `model_conf.json` al subdirectorio adecuado dentro de `TTS/XXX`; encontrarás plantillas en la carpeta `TTS`.
- En el caso de VibeVoice, entra en la carpeta `TTS` y clona el repositorio necesario de Hugging Face, por ejemplo: `git clone https://huggingface.co/sarpba/VibeVoice-large-HUN`.
- Opcionalmente, crea una cuenta de Hugging Face, acepta la licencia de Pyannote Speaker Diarization 3.1 y genera una API key de solo lectura: https://huggingface.co/pyannote/speaker-diarization-3.1
- Registra una cuenta de DeepL, activa el plan gratuito de API y crea una API key propia.
- Registra una cuenta de ElevenLabs para usar el módulo ASR; el plan gratuito suele incluir unas 2 o 3 horas de crédito al mes.

## Módulos del pipeline y catálogo de scripts

El siguiente resumen se basa en los archivos `*_help.md` del directorio `scripts/`. Cada punto indica el entorno conda recomendado; las opciones CLI detalladas se describen en el help correspondiente.

### Preparación de entrada y gestión de archivos
- `scripts/AUDIO-VIDEO/audio_copy_helper/audio-copy-helper.py` (`sync`): convierte audios subidos a WAV de 44,1 kHz y los copia a los subdirectorios del proyecto indicados (`--extracted_audio`, `--separated_audio_*`) usando el parámetro obligatorio `-p/--project-name`.
- `scripts/AUDIO-VIDEO/extract_audio_from_video/extract_audio_easy_channels.py` (`sync`): extrae la pista de audio del primer video de la carpeta `upload`, opcionalmente separa canales con `--keep_channels`, y guarda el resultado en `extracted_audio`.
- `scripts/AUDIO-VIDEO/separate_speak_and_background_audio/separate_audio_easy.py` (`demucs`): separa voz y fondo con modelos Demucs/MDX; se ajusta mediante `--models`, `--chunk_size` y `--background_blend`.
- `scripts/AUDIO-VIDEO/unpack_srt_from_mkv/unpack_srt_from_mkv_easy.py` (`sync`): exporta subtítulos `.srt` incrustados en archivos MKV a la carpeta `subtitles` del proyecto.

### ASR y resegmentación
- `scripts/ASR/canary-non_working_beta/canary-easy.py` (`nemo`): ejecuta NVIDIA Canary ASR sobre los audios de `separated_audio_speech`, con chunks automáticos o fijos y traducción opcional (`--source-lang`, `--target-lang`, `--keep_alternatives`).
- `scripts/ASR/elevenlabs/elevenlabs.py` (`sync`): llama a la API STT de ElevenLabs, escribe JSON normalizado `word_segments` junto a cada archivo y usa `keyholder.json` para la clave API (`--api-key`, `--diarize`).
- `scripts/ASR/paraket-eng/parakeet-tdt-0.6b-v2.py` (`nemo`): ejecuta Parakeet TDT 0.6B v2 con calibración automática de chunks; los límites de segmento se ajustan con `--max-pause`, `--timestamp-padding` y `--max-segment-duration`.
- `scripts/ASR/paraket-multilang/parakeet-tdt-0.6b-v3.py` (`nemo`): pipeline Parakeet TDT v3 multilingüe con la misma CLI y mayor cobertura de idiomas.
- `scripts/ASR/whisperx/whisx_v1.1.py` (`whisperx`): ejecuta WhisperX `large-v3` en varias GPU con diarización pyannote opcional (`--hf_token`, `--gpus`, `--timestamp-padding`) y guarda la salida junto a `separated_audio_speech`.
- `scripts/ASR/resegment/resegment.py` (`sync`): resegmenta el JSON generado por ASR con corrección basada en energía y backup (`--max-pause`, `--timestamp-padding`, `--enforce-single-speaker`).
- `scripts/ASR/resegment-mfa/resegment-mfa.py` (`sync`): usa Montreal Forced Aligner para refinar tiempos de palabra con `--use-mfa-refine`, manteniendo los parámetros habituales de resegmentación.

### Diarización
- `scripts/DIARIZE/speaker_diarize_e2e_non_working_beta/e2e_diarize_speech.py` (`nemo`): diarización end-to-end basada en Sortformer con configuración Hydra y búsqueda opcional de posprocesado con Optuna.
- `scripts/DIARIZE/speaker_diarize_pyannote/split_segments_by_speaker.py` (`sync`): reorganiza segmentos existentes por hablante usando diarización de Pyannote (`--hf-token`, `--min-chunk`, `--no-backup`, `--dry-run`).

### Traducción
- `scripts/TRANSLATE/chatgpt_with_srt/translate_chatgpt_srt_easy.py` (`sync`): traduce con la API de ChatGPT, opcionalmente añade contexto SRT al prompt y guarda la clave en `keyholder.json` (`-input_language`, `-output_language`, `-model`, `-auth_key`).
- `scripts/TRANSLATE/deepl/deepl_translate.py` (`sync`): llama a la API REST de DeepL y escribe en la carpeta `translated`; `--auth-key` se guarda en `keyholder.json` en formato base64, y los códigos de idioma faltantes se completan desde `config.json`.

### TTS
- `scripts/TTS/f5-tts/f5_tts_easy_EQ.py` (`f5-tts`): pipeline base de F5-TTS con EQ y verificación con Whisper (`--norm`, `--eq-config`, `--max-retries`, `--whisper-model`).
- `scripts/TTS/f5-tts-narrator/f5_tts_narrator.py` (`f5-tts`): trabaja desde una carpeta de narrador (`--narrator`) manteniendo los controles multietapa de F5-TTS.
- `scripts/TTS/vibevoice_old/vibevoice-tts.py` (`vibevoice`): TTS VibeVoice base con soporte LoRA, opciones `--norm`, `--cfg_scale`, `--checkpoint_path`, `--seed`, `--max_retries` y verificación posterior con Whisper.
- `scripts/TTS/vibevoice_1.1/vibevoice-tts_v1.1.py` (`vibevoice`): mejora la estabilidad con asignación dedicada de workers GPU y guardado de muestras de referencia para generaciones fallidas.
- `scripts/TTS/vibevoice_1.2/vibevoice-tts_v1.2.py` (`vibevoice`): añade comprobación de pitch y reintentos (`--enable_pitch_check`, `--pitch_tolerance`, `--pitch_retry`) al flujo de VibeVoice.
- `scripts/TTS/vibevoice-narrator/vibevoice-tts_narrator.py` (`vibevoice`): usa una única muestra WAV de narrador (`--narrator`) manteniendo las opciones de EQ y normalización de VibeVoice.

### Mastering de audio y entrega
- `scripts/AUDIO-VIDEO/normalize_and_cut_tts_files_old_version/normalise_and_cut_json_easy.py` (`sync`): ajusta los segmentos generados al RMS de referencia, corta silencios con VAD y opcionalmente elimina archivos vacíos (`--delete_empty`, `--no-sync-loudness`).
- `scripts/AUDIO-VIDEO/normalize_and_cut_tts_files/normalise_and_cut_json_easy_v1.1.py` (`sync`): la versión v1.1 recorta silencio final con `silenceremove` de FFmpeg, aplica límite de pico y mantiene las mismas opciones principales de normalización.
- `scripts/AUDIO-VIDEO/merge_chunks_with_background/merge_chunks_with_background_easy.py` (`sync`): concatena la salida de `translated_splits` según marcas temporales, mezcla fondo en modo narrador con `--background-volume`, y guarda en `film_dubbing`.
- `scripts/AUDIO-VIDEO/merge_chunks_with_bacground_v1.1/merge_chunks_with_background_easy.py` (`sync`): amplía lo anterior con `--time-stretching`, que acelera audio sin alterar el tono para evitar solapamientos.
- `scripts/AUDIO-VIDEO/merge_chunks_with_background_narrator_version/merge_chunks_with_background_beta.py` (`sync`): versión beta enfocada en narrador con `--max-speedup` para controlar la compresión temporal de segmentos superpuestos durante la mezcla de fondo.
- `scripts/AUDIO-VIDEO/merge_dub_to_video/merge_to_video_easy.py` (`sync`): vuelve a muxear el audio final y, si existe, los subtítulos en el video original, etiquetando la pista con `--language` y escribiendo el resultado en `deliveries`.

### Control de calidad y utilidades
- `scripts/AUDIO-VIDEO/pitch_analizer/pitch_analizer.py` (`sync`): compara el pitch de los segmentos `translated_splits` con una referencia de narrador (`--tolerance`, `--delete-outside`) y puede eliminar automáticamente los valores fuera de rango.
- `scripts/NORMALIZER_HELPER/collect_normalized_translations.py` (`f5-tts`): recopila textos normalizados desde JSON de `translated` para el normalizador húngaro, trabajando sobre el proyecto indicado con `-p/--project`.
- `scripts/TTS/EQ.json`: configuración de EQ multibanda consumida por scripts F5-TTS y VibeVoice mediante `--eq-config`; puedes ajustar `global_gain_db` y `points` según tu referencia.

### Documentación para desarrollo
- `scripts/How_to_create_new_script_modul/how_to_create_script_modul.md`: guía resumida, orientada a personas, para crear nuevos módulos de script compatibles con el framework.
- `scripts/How_to_create_new_script_modul/how_to_create_script_modul_AI.md`: nota interna específica del repositorio con las reglas reales de integración para desarrollo asistido por IA/agentes.

### Desarrollo de módulos de script en resumen
- Cada módulo nuevo debe incluir tres archivos: `<name>.py`, `<name>.json` y `<name>_help.md`.
- En los nuevos JSON usa `environment`, no `enviroment`.
- `scripts/scripts.json` es un archivo derivado; no lo edites a mano.
- La lista de parámetros del JSON, `argparse` y `_help.md` debe mantenerse sincronizada 1:1.
- Cada parámetro debe tener un campo `default` explícito.
- Si el script usa API keys o autofill especial, revisa también los mappings correspondientes del backend.

## Ejecución de la aplicación (ES)
```bash
conda activate sync
cd AI_Dubbing
python main_app.py
```

### Enlaces externos

- Docker: [sarpba/ai_dubbing:latest](https://hub.docker.com/r/sarpba/ai_dubbing)
- Runpod: [Template](https://console.runpod.io/deploy?template=5kqz4o0l31&ref=jwyyc9tv)
