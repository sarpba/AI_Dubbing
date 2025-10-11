import os
import argparse
import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
import random
import logging
import importlib.util
import time
import json
import datetime
import tempfile

import soundfile as sf
import tqdm

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything

# Import cached_path (needed for HuggingFace paths if provided)
from cached_path import cached_path

# Logging beállítása
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(processName)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("infer_batch.log")
    ]
)
logger = logging.getLogger(__name__)

# A projekt gyökérkönyvtárának meghatározása a script helyéből
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Model Configuration Constants ---
DEFAULT_MEL_SPEC_TYPE = "vocos"
# --- End Model Configuration Constants ---

class F5TTS:
    def __init__(
        self,
        model_cls,
        model_cfg_dict,
        ckpt_file,
        vocab_file,
        vocoder_name=DEFAULT_MEL_SPEC_TYPE,
        ode_method="euler",
        use_ema=True,
        local_path=None,
        device=None,
    ):
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_cls, model_cfg_dict, ckpt_file, self.mel_spec_type, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name, local_path):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(self, model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema):
        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        file_wave,
        remove_silence=False,
        speed=1.0,
        nfe_step=32,
        seed=-1,
    ):
        if not (0.3 <= speed <= 2.0):
            raise ValueError(f"Invalid speed value: {speed}. Must be between 0.3 and 2.0.")
        if not (16 <= nfe_step <= 64):
            raise ValueError(f"Invalid nfe_step value: {nfe_step}. Must be between 16 and 64.")

        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=logger.info,
            progress=tqdm,
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=nfe_step,
            cfg_strength=2,
            sway_sampling_coef=-1,
            speed=speed,
            fix_duration=None,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        return wav, sr, spect

def time_to_filename_str(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = datetime.timedelta(seconds=seconds)
    minutes, secs = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"

def resolve_model_paths_from_dir(model_dir_path: str):
    model_dir = Path(model_dir_path)
    if not model_dir.is_dir():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    ckpt_files = list(model_dir.glob('*.pt')) + list(model_dir.glob('*.safetensors'))
    if len(ckpt_files) == 0:
        logger.error(f"No checkpoint file (.pt or .safetensors) found in {model_dir}")
        sys.exit(1)
    if len(ckpt_files) > 1:
        logger.error(f"Multiple checkpoint files found in {model_dir}: {ckpt_files}. Please ensure only one is present.")
        sys.exit(1)
    resolved_ckpt_file = ckpt_files[0]

    resolved_vocab_file = model_dir / 'vocab.txt'
    if not resolved_vocab_file.exists():
        logger.error(f"vocab.txt not found in {model_dir}")
        sys.exit(1)

    json_files = list(model_dir.glob('*.json'))
    if len(json_files) == 0:
        logger.error(f"No config file (.json) found in {model_dir}")
        sys.exit(1)
    if len(json_files) > 1:
        logger.error(f"Multiple config files (.json) found in {model_dir}: {json_files}. Please ensure only one is present.")
        sys.exit(1)
    resolved_json_file = json_files[0]

    logger.info(f"Found model files in '{model_dir}':")
    logger.info(f"  - Checkpoint: {resolved_ckpt_file.name}")
    logger.info(f"  - Vocab:      {resolved_vocab_file.name}")
    logger.info(f"  - Config:     {resolved_json_file.name}")

    return str(resolved_ckpt_file), str(resolved_vocab_file), str(resolved_json_file)

def interactive_model_selection(tts_base_path: Path):
    """Lists subdirectories in the TTS folder and prompts the user to select one."""
    logger.info(f"No model directory specified. Searching for models in: {tts_base_path}")
    models = sorted([d for d in tts_base_path.iterdir() if d.is_dir()])
    
    if not models:
        logger.error(f"No models found in the TTS directory: {tts_base_path}")
        sys.exit(1)

    print("\nPlease select a model from the TTS directory:")
    for i, model_path in enumerate(models):
        print(f"  {i + 1}: {model_path.name}")
    
    while True:
        try:
            choice = int(input("Enter number: ")) - 1
            if 0 <= choice < len(models):
                selected_path = models[choice]
                logger.info(f"Model selected: {selected_path.name}")
                return str(selected_path)
            else:
                print("Invalid number, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="F5-TTS Inference Script using a central config.json.")
    
    # Új, pozíciós argumentum a projekt nevéhez
    parser.add_argument("project_name", type=str,
                        help="Name of the project directory inside the main 'workdir'.")
    
    parser.add_argument("--norm", type=str, required=True,
                        help="Normalization type (e.g., 'hun', 'eng').")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Optionally, specify the path to the model directory. If not provided, an interactive selection will be shown.")
    
    # Advanced / optional parameters
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the generated audio (0.3-2.0). Default is 1.0.")
    parser.add_argument("--nfe_step", type=int, default=32, help="Number of NFE steps (16-64). Default is 32.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers. Defaults to the number of available GPUs.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility. Default is -1, which selects a random seed.")
    parser.add_argument("--remove_silence", action="store_true", help="Remove silence from generated audio.")

    args = parser.parse_args()
    return args

def main_worker(gpu, args, all_chunks, input_wav_path_str):
    tasks_chunk = all_chunks[gpu]
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    logger.info(f"Worker {gpu} starting on device {device} with {len(tasks_chunk)} tasks.")

    try:
        normalize_fn = None
        normaliser_path = Path(args.normalisers_dir) / args.norm / "normaliser.py"
        if normaliser_path.exists():
            try:
                spec = importlib.util.spec_from_file_location("normaliser", normaliser_path)
                normaliser = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(normaliser)
                if hasattr(normaliser, 'normalize'):
                    normalize_fn = normaliser.normalize
                    logger.info(f"Worker {gpu}: Loaded normalizer '{args.norm}'.")
            except Exception as e:
                logger.error(f"Worker {gpu}: Failed to load normalizer '{args.norm}': {e}.")
        else:
            logger.error(f"Worker {gpu}: Normalizer module not found at '{normaliser_path}'.")

        architecture_class_map = {"DiT": DiT, "UNetT": UNetT}
        resolved_model_cls = architecture_class_map.get(args.resolved_model_architecture)
        if resolved_model_cls is None:
            logger.error(f"Worker {gpu}: Unknown model architecture '{args.resolved_model_architecture}'.")
            sys.exit(1)

        f5tts = F5TTS(
            model_cls=resolved_model_cls,
            model_cfg_dict=args.resolved_model_params_dict,
            ckpt_file=args.resolved_ckpt_file,
            vocab_file=args.resolved_vocab_file,
            device=device
        )
        logger.info(f"Worker {gpu}: Initialized '{args.resolved_model_architecture}' model on device {device}.")

        full_audio_data, sample_rate = sf.read(input_wav_path_str)
        logger.info(f"Worker {gpu}: Loaded main audio file {Path(input_wav_path_str).name}")

        for segment in tqdm.tqdm(tasks_chunk, desc=f"Processing on {device}"):
            start_time = segment.get('start')
            end_time = segment.get('end')
            ref_text = segment.get('text', '').strip()
            gen_text = segment.get('translated_text', '').strip()

            if not all([isinstance(start_time, (int, float)), isinstance(end_time, (int, float)), ref_text, gen_text]):
                continue

            filename = f"{time_to_filename_str(start_time)}_{time_to_filename_str(end_time)}.wav"
            output_wav_path = Path(args.output_dir) / filename
            if output_wav_path.exists():
                continue

            if normalize_fn:
                try:
                    gen_text = normalize_fn(gen_text)
                except Exception as e:
                    logger.error(f"Worker {gpu}: Normalization failed for segment {start_time}-{end_time}: {e}")
                    continue
            
            try:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                if end_sample > len(full_audio_data): end_sample = len(full_audio_data)
                if start_sample >= end_sample: continue
                ref_audio_chunk = full_audio_data[start_sample:end_sample]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True, mode='w+b') as tmp_ref_file:
                    sf.write(tmp_ref_file.name, ref_audio_chunk, sample_rate)
                    f5tts.infer(
                        ref_file=tmp_ref_file.name, ref_text=ref_text, gen_text=gen_text,
                        file_wave=str(output_wav_path), remove_silence=args.remove_silence,
                        speed=args.speed, nfe_step=args.nfe_step, seed=args.seed,
                    )
            except Exception as e:
                logger.error(f"Worker {gpu}: Error processing segment {start_time}-{end_time}: {e}", exc_info=True)
                continue
    except Exception as e:
        logger.critical(f"Critical error in worker {gpu} on device {device}: {e}", exc_info=True)

def process_file_pair(args, input_wav_path, input_json_path):
    logger.info(f"Starting processing for WAV: '{input_wav_path.name}' and JSON: '{input_json_path.name}'")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            segments = data.get("segments", [])
            if not segments:
                logger.error(f"No 'segments' found in the JSON file: {input_json_path.name}")
                return
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON file {input_json_path.name}: {e}")
            return

    segments.sort(key=lambda x: x.get('start', 0))

    logger.info("Loading audio file for noise extraction...")
    try:
        full_audio_data, sample_rate = sf.read(input_wav_path)
        duration_seconds = len(full_audio_data) / sample_rate
    except Exception as e:
        logger.error(f"Failed to load audio file {input_wav_path.name}: {e}")
        return
    logger.info(f"Audio loaded. Duration: {duration_seconds:.2f}s. Sample rate: {sample_rate}Hz.")

    logger.info("Extracting non-segmented (noise) parts...")
    last_end_time = 0.0
    for segment in segments:
        start_time = segment.get('start')
        if start_time is None: continue
        if start_time > last_end_time:
            noise_filename = f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(start_time)}.wav"
            noise_output_path = Path(args.output_dir_noise) / noise_filename
            if not noise_output_path.exists():
                start_sample, end_sample = int(last_end_time * sample_rate), int(start_time * sample_rate)
                sf.write(noise_output_path, full_audio_data[start_sample:end_sample], sample_rate)
        last_end_time = segment.get('end', start_time)

    if duration_seconds > last_end_time:
        noise_filename = f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(duration_seconds)}.wav"
        noise_output_path = Path(args.output_dir_noise) / noise_filename
        if not noise_output_path.exists():
            start_sample = int(last_end_time * sample_rate)
            sf.write(noise_output_path, full_audio_data[start_sample:], sample_rate)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("No GPUs detected. This script requires at least one GPU. Exiting.")
        sys.exit(1)
    
    max_workers = args.max_workers if args.max_workers is not None else num_gpus
    num_workers = min(num_gpus, max_workers)
    if num_workers <= 0:
        logger.error("Number of workers must be positive. Exiting.")
        sys.exit(1)
    logger.info(f"Using {num_workers} GPU(s) for TTS generation.")

    chunks = [[] for _ in range(num_workers)]
    for idx, segment in enumerate(segments):
        chunks[idx % num_workers].append(segment)

    logger.info("Starting TTS generation workers...")
    spawn_args = (args, chunks, str(input_wav_path))
    mp.spawn(main_worker, nprocs=num_workers, args=spawn_args)
    logger.info(f"Finished TTS generation for {input_wav_path.name}.")

def main():
    # --- 1. Load Central Configuration ---
    config_path = PROJECT_ROOT / 'config.json'
    if not config_path.exists():
        logger.error(f"Central config file not found at: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    cfg_dirs = config_data['DIRECTORIES']
    cfg_subdirs = config_data['PROJECT_SUBDIRS']

    # --- 2. Parse Command-Line Arguments ---
    args = parse_arguments()

    # --- 3. Construct Project Paths from config and project_name ---
    workdir_path = PROJECT_ROOT / cfg_dirs['workdir']
    full_project_path = workdir_path / args.project_name
    
    if not full_project_path.is_dir():
        logger.error(f"Project directory not found: {full_project_path}")
        sys.exit(1)
    
    input_wav_dir = full_project_path / cfg_subdirs['separated_audio_speech']
    input_json_dir = full_project_path / cfg_subdirs['translated']
    output_dir = full_project_path / cfg_subdirs['translated_splits']
    output_dir_noise = full_project_path / cfg_subdirs['noice_splits']
    
    # Create output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_noise.mkdir(parents=True, exist_ok=True)
    
    # Add resolved paths to args for worker processes
    args.output_dir = str(output_dir)
    args.output_dir_noise = str(output_dir_noise)
    args.normalisers_dir = str(PROJECT_ROOT / cfg_dirs['normalisers'])

    # --- 4. Determine Model Directory ---
    if args.model_dir:
        model_dir_path = args.model_dir
    else:
        tts_base_path = PROJECT_ROOT / cfg_dirs['TTS']
        model_dir_path = interactive_model_selection(tts_base_path)

    # --- 5. Resolve Model Files and Config ---
    ckpt_path, vocab_path, json_path = resolve_model_paths_from_dir(model_dir_path)
    args.resolved_ckpt_file = ckpt_path
    args.resolved_vocab_file = vocab_path
    
    with open(json_path, 'r', encoding='utf-8') as f:
        model_config_data = json.load(f)

    if "model_architecture" not in model_config_data or "model_params" not in model_config_data:
        logger.error(f"Model config '{json_path}' is missing 'model_architecture' or 'model_params' keys.")
        sys.exit(1)

    args.resolved_model_architecture = model_config_data["model_architecture"]
    args.resolved_model_params_dict = model_config_data["model_params"]

    # --- 6. Find input wav/json files ---
    wav_files = list(input_wav_dir.glob('*.wav'))
    json_files = list(input_json_dir.glob('*.json'))

    if not wav_files:
        logger.error(f"No .wav files found in directory: {input_wav_dir}")
        sys.exit(1)
    if len(wav_files) > 1:
        logger.warning(f"Multiple .wav files found in {input_wav_dir}. Using the first one: {wav_files[0]}")
    
    if not json_files:
        logger.error(f"No .json files found in directory: {input_json_dir}")
        sys.exit(1)
    if len(json_files) > 1:
        logger.warning(f"Multiple .json files found in {input_json_dir}. Using the first one: {json_files[0]}")

    input_wav_path = wav_files[0]
    input_json_path = json_files[0]

    # --- 7. Start Processing ---
    process_file_pair(args, input_wav_path, input_json_path)
    logger.info("Script finished.")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()