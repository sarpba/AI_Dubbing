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

# Maximum number of retries for processing a single file
MAX_RETRIES = 3
# Timeout in seconds
TIMEOUT = 600  # 10 minutes

def get_available_gpus():
    """
    Queries the available GPU indices using nvidia-smi.
    """
    try:
        command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_indices = result.stdout.decode().strip().split('\n')
        gpu_ids = [int(idx) for idx in gpu_indices if idx.strip().isdigit()]
        return gpu_ids
    except Exception as e:
        print(f"Error querying GPUs: {e}")
        return []

# This function returns the duration of the audio file in seconds.
def get_audio_duration(audio_file):
    command = [
        "ffprobe",
        "-i", audio_file,
        "-show_entries", "format=duration",
        "-v", "quiet",
        "-of", "csv=p=0"
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_str = result.stdout.decode().strip()
        duration = float(duration_str)
        return duration
    except Exception as e:
        print(f"Failed to determine audio length for {audio_file}: {e}")
        return 0

# This function handles processes assigned to each GPU.
def worker(gpu_id, task_queue, hf_token, language_code):
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

        output_dir = os.path.dirname(audio_file)

        try:
            print(f"Processing on GPU-{gpu_id}: {audio_file}")
            start_time = time.time()
            start_datetime = datetime.datetime.now()

            # 1. Load the WhisperX model
            if language_code:
                model = whisperx.load_model("large-v3", device=device, compute_type="float16", language=language_code)
            else:
                model = whisperx.load_model("large-v3", device=device, compute_type="float16")

            # 2. Load and transcribe the audio
            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=16)
            print(f"Transcription completed for {audio_file}")

            # Determine the language code for alignment
            align_language_code = language_code if language_code else result["language"]

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
                    print("Need HF_token for diarization")
            else:
                print("Need HF_token for diarization")

            # 5. Save the results
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            # 6. Clean up to free GPU memory
            del model
            del model_a
            if hf_token:
                del diarize_model
            gc.collect()
            torch.cuda.empty_cache()

            end_time = time.time()
            end_datetime = datetime.datetime.now()
            processing_time = end_time - start_time

            audio_duration = get_audio_duration(audio_file)
            ratio = audio_duration / processing_time if processing_time > 0 else 0

            print(f"Successfully processed on GPU-{gpu_id}:")
            print(f"Processed file: {audio_file},")
            print(f"Audio length: {audio_duration:.2f} s,")
            print(f"Processing time: {processing_time:.2f} s,")
            print(f"Ratio: {ratio:.2f}")
            print(f"Start time: {start_datetime.strftime('%Y.%m.%d %H:%M')}")
            print(f"End time: {end_datetime.strftime('%Y.%m.%d %H:%M')}\n")

        except Exception as e:
            print(f"Error processing {audio_file} on GPU-{gpu_id}: {e}")
            if retries < MAX_RETRIES:
                print(f"Retrying {retries + 1}/{MAX_RETRIES}...")
                task_queue.put((audio_file, retries + 1))
            else:
                print(f"Maximum retries reached: Processing {audio_file} failed.\n")

# This function collects all audio files in the given directory and its subdirectories.
def get_audio_files(directory):
    audio_extensions = (".mp3", ".wav", ".flac", ".m4a", ".opus")
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

# This function starts the processes assigned to GPUs and manages the task list.
def transcribe_directory(directory, gpu_ids, hf_token, language_code):
    audio_files = get_audio_files(directory)
    task_queue = Queue()
    tasks_added = 0
    for audio_file in audio_files:
        json_file = os.path.splitext(audio_file)[0] + ".json"
        if not os.path.exists(json_file):
            task_queue.put((audio_file, 0))
            tasks_added += 1
        else:
            print(f"Already exists: {json_file}, skipping in task list...")

    if tasks_added == 0:
        print("No files to process.")
        return

    processes = []
    for gpu_id in gpu_ids:
        p = Process(target=worker, args=(gpu_id, task_queue, hf_token, language_code))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def load_config_and_get_paths(project_name: str) -> str:
    """Betölti a config.json-t, és visszaadja a projekt feldolgozandó mappájának elérési útját."""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
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
    parser = argparse.ArgumentParser(description="Transcribe audio files in a project-specific directory (config.json alapú) WhisperX és több GPU használatával.")
    parser.add_argument("-p", "--project-name", required=True, help="A projekt neve (a 'workdir' alatti mappa), amit fel kell dolgozni.")
    parser.add_argument('--gpus', type=str, default=None, help="GPU indices to use, separated by commas (e.g., '0,2,3')")
    parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models. If not provided, diarization will be skipped.")
    parser.add_argument('--language', type=str, default=None, help="Optional language code (e.g., 'en', 'es'). If not provided, language detection is used.")

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
            print("Error: No GPUs available.")
            sys.exit(1)
        invalid_gpus = [gpu for gpu in specified_gpus if gpu not in available_gpus]
        if invalid_gpus:
            print(f"Error: The specified GPUs are not available: {invalid_gpus}")
            sys.exit(1)
        gpu_ids = specified_gpus
    else:
        gpu_ids = get_available_gpus()
        if not gpu_ids:
            print("Error: No GPUs available.")
            sys.exit(1)

    # Start transcription with the determined GPUs and optional language code
    transcribe_directory(processing_directory, gpu_ids, args.hf_token, args.language)
