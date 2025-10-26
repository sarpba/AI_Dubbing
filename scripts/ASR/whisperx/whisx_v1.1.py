import os
import json
import argparse
import whisperx
import gc
import time
import datetime
import sys
from multiprocessing import Process, Queue
from pathlib import Path
import torch
import subprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_project_root() -> Path:
    """
    Felkeresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")

# Maximum number of retries for processing a single file
MAX_RETRIES = 3
# Timeout in seconds
TIMEOUT = 600  # 10 minutes

# ==============================================================================
#   SZEGMENTÁLÁSI LOGIKA (A PARAKEET-TDT-0.6B-V2 SZKRIPTBŐL ÁTVÉVE)
# ==============================================================================

def adjust_word_timestamps(word_segments, padding_s: float):
    """Kismértékben kiterjeszti a szó időbélyegeket, hogy kitöltse a kisebb szüneteket."""
    if not word_segments or padding_s <= 0: return word_segments
    adjusted_segments = [word.copy() for word in word_segments]
    for i in range(len(adjusted_segments) - 1):
        gap = adjusted_segments[i+1]['start'] - adjusted_segments[i]['end']
        if gap > (padding_s * 2):
            adjustment = min(padding_s, gap / 2.0)
            adjusted_segments[i]['end'] += adjustment
            adjusted_segments[i+1]['start'] -= adjustment
    adjusted_segments[0]['start'] = max(0.0, adjusted_segments[0]['start'] - padding_s)
    adjusted_segments[-1]['end'] += padding_s
    for word in adjusted_segments:
        word['start'] = round(word['start'], 3)
        word['end'] = round(word['end'], 3)
    return adjusted_segments

def _create_segment_from_words(words):
    """Létrehoz egyetlen szegmens objektumot egy szavakból álló listából."""
    if not words: return None
    text = " ".join(w['word'] for w in words)
    text = text.replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return {"start": words[0]['start'], "end": words[-1]['end'], "text": text, "words": words}

def sentence_segments_from_words(words, max_pause_s: float):
    """Szavak listáját mondatszegmensekre bontja a szavak közötti szünetek alapján."""
    segs, current = [], []
    start_time = None
    for w in words:
        if current and (w["start"] - current[-1]["end"] > max_pause_s):
            if new_seg := _create_segment_from_words(current): segs.append(new_seg)
            current, start_time = [], None
        if start_time is None: start_time = w["start"]
        current.append(w)
        token = w["word"].strip()
        if token in {".", "!", "?"} or token[-1] in {".", "!", "?"}:
            if new_seg := _create_segment_from_words(current): segs.append(new_seg)
            current, start_time = [], None
    if new_seg := _create_segment_from_words(current): segs.append(new_seg)
    return segs

def split_long_segments(segments, max_duration_s: float):
    """A túl hosszú szegmenseket feldarabolja a megadott maximális időtartam szerint."""
    if max_duration_s <= 0: return segments
    final_segments = []
    for segment in segments:
        words_to_process = segment['words']
        while words_to_process:
            segment_duration = words_to_process[-1]['end'] - words_to_process[0]['start']
            if segment_duration <= max_duration_s:
                if new_seg := _create_segment_from_words(words_to_process): final_segments.append(new_seg)
                break
            else:
                candidate_words = [w for w in words_to_process if w['end'] - words_to_process[0]['start'] <= max_duration_s]
                if not candidate_words: candidate_words = words_to_process[:1]

                best_split_idx = len(candidate_words) - 1
                if len(candidate_words) > 1:
                    max_gap = -1.0
                    for i in range(len(candidate_words) - 1):
                        gap = candidate_words[i+1]['start'] - candidate_words[i]['end']
                        if gap >= max_gap:
                            max_gap = gap
                            best_split_idx = i
                
                new_segment_words = words_to_process[:best_split_idx + 1]
                if new_seg := _create_segment_from_words(new_segment_words): final_segments.append(new_seg)
                words_to_process = words_to_process[best_split_idx + 1:]
    return final_segments

# ==============================================================================
#   EREDETI WHISX.PY FÜGGVÉNYEK
# ==============================================================================

def get_available_gpus():
    """Queries the available GPU indices using nvidia-smi."""
    try:
        command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_indices = result.stdout.decode().strip().split('\n')
        gpu_ids = [int(idx) for idx in gpu_indices if idx.strip().isdigit()]
        return gpu_ids
    except Exception as e:
        print(f"Error querying GPUs: {e}")
        return []

def get_audio_duration(audio_file):
    """This function returns the duration of the audio file in seconds."""
    command = [
        "ffprobe", "-i", audio_file, "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0"
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_str = result.stdout.decode().strip()
        duration = float(duration_str)
        return duration
    except Exception as e:
        print(f"Failed to determine audio length for {audio_file}: {e}")
        return 0

def worker(gpu_id, task_queue, hf_token, language_code, max_pause_s, padding_s, max_segment_s):
    """This function handles processes assigned to each GPU."""
    device = "cuda"
    try:
        torch.cuda.set_device(gpu_id)
    except Exception as e:
        print(f"Error setting CUDA device to {gpu_id}: {e}")
        return

    while True:
        try:
            task = task_queue.get_nowait()
            audio_file, retries = task
        except Exception:
            break

        json_file = os.path.splitext(audio_file)[0] + ".json"
        if os.path.exists(json_file):
            print(f"Already exists: {json_file}, skipping in worker...")
            continue

        try:
            print(f"Processing on GPU-{gpu_id}: {audio_file}")
            start_time = time.time()
            start_datetime = datetime.datetime.now()

            # 1. Load the WhisperX model
            model = whisperx.load_model("large-v3", device, compute_type="float16", language=language_code)

            # 2. Load and transcribe the audio
            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=16)
            
            # <<< JAVÍTÁS: Mentsük el a nyelvet, mielőtt az align felülírja a result objektumot.
            detected_language = result["language"]
            print(f"Transcription completed for {audio_file}. Detected language: {detected_language}")

            # Determine the language code for alignment
            align_language_code = language_code if language_code else detected_language

            # 3. Align the transcription
            model_a, metadata = whisperx.load_align_model(language_code=align_language_code, device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            print(f"Alignment completed for {audio_file}")

            # 4. Perform diarization if hf_token is provided
            if hf_token:
                try:
                    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                    diarize_segments = diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    print(f"Diarization completed for {audio_file}")
                except Exception as dia_e:
                    print(f"Error during diarization for {audio_file}: {dia_e}")
            else:
                print("Skipping diarization (no HF token).")

            # --- START: NEW CUSTOM SEGMENTATION LOGIC ---
            print(f"Applying custom segmentation for {audio_file}...")
            all_words = []
            if 'segments' in result and result['segments']:
                for segment in result['segments']:
                    if 'words' in segment and segment['words']:
                        all_words.extend(segment['words'])
            
            if not all_words:
                print(f"Warning: No word segments found after alignment for {audio_file}. Saving original result.")
                result_to_save = result
            else:
                adjusted_words = adjust_word_timestamps(all_words, padding_s=padding_s)
                initial_segments = sentence_segments_from_words(adjusted_words, max_pause_s=max_pause_s)
                final_segments = split_long_segments(initial_segments, max_duration_s=max_segment_s)
                
                # <<< JAVÍTÁS: Az elmentett `detected_language` változót használjuk.
                result_to_save = {
                    "language": detected_language,
                    "segments": final_segments,
                    "word_segments": adjusted_words
                }
                print(f"  - {len(final_segments)} new sentence segments generated (max_pause={max_pause_s}s, padding={padding_s}s, max_dur={max_segment_s}s).")
            # --- END: NEW CUSTOM SEGMENTATION LOGIC ---

            # 5. Save the results
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result_to_save, f, ensure_ascii=False, indent=4)

            # 6. Clean up to free GPU memory
            del model
            del model_a
            if 'diarize_model' in locals():
                del diarize_model
            gc.collect()
            torch.cuda.empty_cache()

            end_time = time.time()
            end_datetime = datetime.datetime.now()
            processing_time = end_time - start_time
            audio_duration = get_audio_duration(audio_file)
            ratio = audio_duration / processing_time if processing_time > 0 else 0

            print(f"Successfully processed on GPU-{gpu_id}:")
            print(f"  - Processed file: {audio_file},")
            print(f"  - Audio length: {audio_duration:.2f} s,")
            print(f"  - Processing time: {processing_time:.2f} s,")
            print(f"  - Ratio: {ratio:.2f}")
            print(f"  - Start time: {start_datetime.strftime('%Y.%m.%d %H:%M')}")
            print(f"  - End time: {end_datetime.strftime('%Y.%m.%d %H:%M')}\n")

        except Exception as e:
            print(f"Error processing {audio_file} on GPU-{gpu_id}: {e}")
            if retries < MAX_RETRIES:
                print(f"Retrying {retries + 1}/{MAX_RETRIES}...")
                task_queue.put((audio_file, retries + 1))
            else:
                print(f"Maximum retries reached: Processing {audio_file} failed.\n")

def get_audio_files(directory):
    """This function collects all audio files in the given directory and its subdirectories."""
    audio_extensions = (".mp3", ".wav", ".flac", ".m4a", ".opus")
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def transcribe_directory(directory, gpu_ids, hf_token, language_code, args):
    """This function starts the processes assigned to GPUs and manages the task list."""
    audio_files = get_audio_files(directory)
    task_queue = Queue()
    tasks_added = 0
    for audio_file in audio_files:
        json_file = os.path.splitext(audio_file)[0] + ".json"
        if not os.path.exists(json_file):
            task_queue.put((audio_file, 0))
            tasks_added += 1
        else:
            print(f"Already exists: {json_file}, skipping task...")

    if tasks_added == 0:
        print("No new files to process.")
        return
    
    print(f"\nStarting processing for {tasks_added} file(s) on {len(gpu_ids)} GPU(s)...")

    processes = []
    for gpu_id in gpu_ids:
        p = Process(target=worker, args=(
            gpu_id, task_queue, hf_token, language_code,
            args.max_pause, args.timestamp_padding, args.max_segment_duration
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def load_config_and_get_paths(project_name: str) -> str:
    """Betölti a config.json-t, és visszaadja a projekt feldolgozandó mappájának elérési útját."""
    try:
        project_root = get_project_root()
        config_path = project_root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"A 'config.json' nem található a projekt gyökerében: {project_root}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        input_subdir = config["PROJECT_SUBDIRS"]["separated_audio_speech"]
        processing_path = workdir / project_name / input_subdir
        if not processing_path.is_dir():
            raise FileNotFoundError(f"A feldolgozandó mappa nem létezik: {processing_path}")
        print("Projekt beállítások betöltve:")
        print(f"  - Projekt név:     {project_name}")
        print(f"  - Feldolgozandó mappa (bemenet és kimenet): {processing_path}")
        return str(processing_path)
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Hiba a konfiguráció betöltése vagy az útvonalak meghatározása közben: {e}")
        print("Kérlek, ellenőrizd a 'config.json' fájlt és a projekt mappaszerkezetét.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files with WhisperX, using custom segmentation.")
    parser.add_argument("-p", "--project-name", required=True, help="A projekt neve (a 'workdir' alatti mappa), amit fel kell dolgozni.")
    parser.add_argument('--gpus', type=str, default=None, help="GPU indices to use, separated by commas (e.g., '0,2,3')")
    parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face Access Token for diarization. If not provided, diarization is skipped.")
    parser.add_argument('--language', type=str, default=None, help="Language code (e.g., 'en', 'es'). If not provided, language detection is used.")
    
    # --- Arguments for custom segmentation ---
    parser.add_argument("--max-pause", type=float, default=0.6, help="Mondatok közti maximális szünet másodpercben (alapért.: 0.6s).")
    parser.add_argument("--timestamp-padding", type=float, default=0.2, help="Időbélyegek kiterjesztése a szünetek rovására (mp). 0 a kikapcsoláshoz (alapért.: 0.2s).")
    parser.add_argument("--max-segment-duration", type=float, default=11.5, help="Mondatszegmensek maximális hossza másodpercben (alapért.: 11.5s). 0 a kikapcsoláshoz.")

    args = parser.parse_args()

    processing_directory = load_config_and_get_paths(args.project_name)

    # Determine GPUs to use
    if args.gpus:
        try:
            specified_gpus = [int(x.strip()) for x in args.gpus.split(',')]
        except ValueError:
            print("Error: The --gpus argument must be a comma-separated list of integers.")
            sys.exit(1)
        available_gpus = get_available_gpus()
        if not available_gpus:
            print("Error: No GPUs available according to nvidia-smi.")
            sys.exit(1)
        invalid_gpus = [gpu for gpu in specified_gpus if gpu not in available_gpus]
        if invalid_gpus:
            print(f"Error: The specified GPUs are not available: {invalid_gpus}. Available GPUs: {available_gpus}")
            sys.exit(1)
        gpu_ids = specified_gpus
    else:
        gpu_ids = get_available_gpus()
        if not gpu_ids:
            print("Error: No GPUs available. Please check your NVIDIA driver installation.")
            sys.exit(1)

    # Start transcription with the determined GPUs and optional language code
    transcribe_directory(processing_directory, gpu_ids, args.hf_token, args.language, args)
