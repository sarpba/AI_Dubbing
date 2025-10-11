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

# Importok az f5_tts könyvtárból
from f5_tts.infer.utils_infer import (
    hop_length, # Ezt felülírjuk a configból
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    target_sample_rate, # Ezt felülírjuk a configból
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything

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

# A projekt gyökérkönyvtárának meghatározása
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class F5TTS:
    def __init__(
        self,
        model_config: dict, # Az egész modell konfigurációs dict
        ckpt_file: str,
        vocab_file: str,
        ode_method: str = "euler",
        use_ema: bool = True,
        device: str = None,
    ):
        self.final_wave = None
        self.model_config = model_config # Mentjük a configot az osztályban

        # Paraméterek kinyerése a configból
        self.target_sample_rate = self.model_config["model"]["mel_spec"]["target_sample_rate"]
        self.hop_length = self.model_config["model"]["mel_spec"]["hop_length"]
        self.mel_spec_type = self.model_config["model"]["mel_spec"]["mel_spec_type"]

        self.seed = -1

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Vocoder betöltése a configból származó adatokkal
        self.load_vocoder_model(self.model_config["model"]["vocoder"])
        # EMA modell betöltése a configból származó adatokkal
        self.load_ema_model(
            self.model_config["model"], # Átadjuk a 'model' szekciót
            ckpt_file,
            vocab_file,
            ode_method,
            use_ema
        )

    def load_vocoder_model(self, vocoder_cfg: dict):
        """
        Vocoder modell betöltése a konfigurációs adatok alapján.
        """
        is_local = vocoder_cfg.get("is_local", False)
        local_path = vocoder_cfg.get("local_path", None)
        
        logger.info(f"Loading vocoder: {self.mel_spec_type}, is_local: {is_local}, local_path: {local_path}")
        self.vocoder = load_vocoder(
            self.mel_spec_type,
            is_local,
            local_path,
            self.device
        )
        logger.info("Vocoder loaded successfully.")

    def load_ema_model(self, model_section_cfg: dict, ckpt_file: str, vocab_file: str, ode_method: str, use_ema: bool):
        """
        EMA (Exponential Moving Average) modell betöltése a konfigurációs adatok alapján.
        """
        model_backbone = model_section_cfg["backbone"]
        model_arch_cfg = model_section_cfg["arch"] # A teljes 'arch' dict

        if model_backbone == "DiT":
            model_cls = DiT
        elif model_backbone == "UNetT":
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model backbone in config: {model_backbone}")

        logger.info(f"Loading EMA model: {model_backbone} with arch config: {model_arch_cfg}")
        self.ema_model = load_model(
            model_cls,
            model_arch_cfg, # Itt adjuk át a configból származó architektúra dict-et
            ckpt_file,
            self.mel_spec_type, # Ezt a konstruktorból kapjuk, a model_config alapján
            vocab_file,
            ode_method,
            use_ema,
            self.device
        )
        logger.info("EMA model loaded successfully.")

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="F5-TTS Inference Script based on JSON segmentation from directories.")
    parser.add_argument("--input_wav_dir", type=str, required=True, help="Path to the directory containing the input .wav file.")
    parser.add_argument("--input_json_dir", type=str, required=True, help="Path to the directory containing the JSON file with segmentation.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory to save generated TTS .wav files.")
    parser.add_argument("--output_dir_noise", type=str, required=True, help="Output directory to save non-segmented (noise/in-between) audio parts.")
    parser.add_argument("--remove_silence", action="store_true", help="Remove silence from generated audio.")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to the vocabulary file (e.g., /path/to/vocab.txt)")
    parser.add_argument("--ckpt_file", type=str, required=True, help="Path to the model checkpoint file (e.g., /path/to/model.pt)")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the model configuration JSON file (e.g., /path/to/config.json)") # ÚJ ARGUMENTUM
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the generated audio (0.3-2.0). Default is 1.0.")
    parser.add_argument("--nfe_step", type=int, default=32, help="Number of NFE steps (16-64). Default is 32.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers. Defaults to the number of available GPUs.")
    parser.add_argument("--norm", type=str, required=False, default=None, help="Normalization type (e.g., 'hun', 'eng'). Determines which normalizer to use.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility. Default is -1, which selects a random seed.")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Konfigurációs fájl betöltése JSON formátumban."""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {config_file}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON config file {config_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading config {config_file}: {e}")

def main_worker(gpu, args, all_chunks, input_wav_path_str, model_config): # model_config paraméter hozzáadva
    # A worker a saját GPU azonosítója alapján kiválasztja a neki szánt feladatcsomagot
    tasks_chunk = all_chunks[gpu]

    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    logger.info(f"Worker {gpu} starting on device {device} with {len(tasks_chunk)} tasks.")

    try:
        normalize_fn = None
        if args.norm is not None:
            normaliser_path = PROJECT_ROOT / "normalisers" / args.norm / "normaliser.py"
            if normaliser_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("normaliser", normaliser_path)
                    normaliser = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(normaliser)
                    if hasattr(normaliser, 'normalize'):
                        normalize_fn = normaliser.normalize
                        logger.info(f"Worker {gpu}: Loaded normalizer '{args.norm}'.")
                    else:
                        logger.error(f"Worker {gpu}: Normalizer module '{normaliser_path}' lacks 'normalize' function.")
                except Exception as e:
                    logger.error(f"Worker {gpu}: Failed to load normalizer '{args.norm}': {e}.")
            else:
                logger.error(f"Worker {gpu}: Normalizer module not found for norm='{args.norm}'.")

        f5tts = F5TTS(
            model_config=model_config, # Itt adjuk át a betöltött configot
            vocab_file=args.vocab_file,
            ckpt_file=args.ckpt_file,
            device=device
        )
        logger.info(f"Worker {gpu}: Initialized F5TTS on device {device}")

        full_audio_data, sample_rate = sf.read(input_wav_path_str)
        logger.info(f"Worker {gpu}: Loaded main audio file {Path(input_wav_path_str).name}")

        # JAVÍTÁS: A ciklus most már a helyes, kiosztott feladatlistán ('tasks_chunk') iterál
        for segment in tqdm.tqdm(tasks_chunk, desc=f"Processing on {device}"):
            start_time = segment.get('start')
            end_time = segment.get('end')
            ref_text = segment.get('text', '').strip()
            gen_text = segment.get('translated_text', '').strip()

            if not all([isinstance(start_time, (int, float)), isinstance(end_time, (int, float)), ref_text, gen_text]):
                logger.warning(f"Worker {gpu}: Skipping malformed segment: {segment}")
                continue

            filename = f"{time_to_filename_str(start_time)}_{time_to_filename_str(end_time)}.wav"
            output_wav_path = Path(args.output_dir) / filename

            if output_wav_path.exists():
                logger.info(f"Worker {gpu}: Output file {output_wav_path} already exists. Skipping.")
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
                
                if end_sample > len(full_audio_data):
                    end_sample = len(full_audio_data)
                if start_sample >= end_sample:
                    logger.warning(f"Worker {gpu}: Invalid segment {start_time}-{end_time}, start is after end. Skipping.")
                    continue

                ref_audio_chunk = full_audio_data[start_sample:end_sample]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True, mode='w+b') as tmp_ref_file:
                    sf.write(tmp_ref_file.name, ref_audio_chunk, sample_rate)
                    f5tts.infer(
                        ref_file=tmp_ref_file.name,
                        ref_text=ref_text,
                        gen_text=gen_text,
                        file_wave=str(output_wav_path),
                        remove_silence=args.remove_silence,
                        speed=args.speed,
                        nfe_step=args.nfe_step,
                        seed=args.seed,
                    )
                logger.info(f"Worker {gpu}: Generated audio saved to {output_wav_path}")

            except Exception as e:
                logger.error(f"Worker {gpu}: Error processing segment {start_time}-{end_time}: {e}", exc_info=True)
                continue

    except Exception as e:
        logger.critical(f"Critical error in worker {gpu} on device {device}: {e}", exc_info=True)

def process_file_pair(args, input_wav_path, input_json_path, model_config): # model_config paraméter hozzáadva
    logger.info(f"Starting processing for WAV: '{input_wav_path.name}' and JSON: '{input_json_path.name}'")
    
    output_dir = Path(args.output_dir)
    output_dir_noise = Path(args.output_dir_noise)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_noise.mkdir(parents=True, exist_ok=True)

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
            noise_output_path = output_dir_noise / noise_filename
            if not noise_output_path.exists():
                start_sample, end_sample = int(last_end_time * sample_rate), int(start_time * sample_rate)
                sf.write(noise_output_path, full_audio_data[start_sample:end_sample], sample_rate)
                logger.info(f"Saved noise segment to {noise_output_path}")
        last_end_time = segment.get('end', start_time)

    if duration_seconds > last_end_time:
        noise_filename = f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(duration_seconds)}.wav"
        noise_output_path = output_dir_noise / noise_filename
        if not noise_output_path.exists():
            start_sample = int(last_end_time * sample_rate)
            sf.write(noise_output_path, full_audio_data[start_sample:], sample_rate)
            logger.info(f"Saved final noise segment to {noise_output_path}")

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
    spawn_args = (args, chunks, str(input_wav_path), model_config) # model_config hozzáadva a worker argumentumaihoz
    mp.spawn(main_worker, nprocs=num_workers, args=spawn_args)
    logger.info(f"Finished TTS generation for {input_wav_path.name}.")

def main():
    args = parse_arguments()
    logger.info(f"Project root determined as: {PROJECT_ROOT}")

    # Konfigurációs fájl betöltése
    try:
        model_config = load_config(args.config_file)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.critical(f"Failed to load model configuration: {e}")
        sys.exit(1)

    input_wav_dir = Path(args.input_wav_dir)
    input_json_dir = Path(args.input_json_dir)

    if not (0.3 <= args.speed <= 2.0):
        logger.error(f"Invalid speed value: {args.speed}. Must be between 0.3 and 2.0.")
        sys.exit(1)
    if not (16 <= args.nfe_step <= 64):
        logger.error(f"Invalid nfe_step value: {args.nfe_step}. Must be between 16 and 64.")
        sys.exit(1)

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

    process_file_pair(args, input_wav_path, input_json_path, model_config) # model_config átadása

    logger.info("Script finished.")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()