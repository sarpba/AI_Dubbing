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
logger.info(f"Project root determined as: {PROJECT_ROOT}")

class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
    ):
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        # Eszköz beállítása: ha nem adunk meg explicit device-t, akkor a CUDA-t, vagy mps-t, illetve CPU-t választja.
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Modellek betöltése
        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name, local_path):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

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

        # A preprocess_ref_audio_text függvény nem várja a device argumentumot, ezért eltávolítjuk azt.
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="F5-TTS Batch Inference Script with Multi-GPU Support")
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True,
        help="Input directory containing .wav and corresponding .txt files (reference texts)"
    )
    parser.add_argument(
        "-ig", "--input_gen_dir", type=str, required=True,
        help="Input generation directory containing .txt files (generated texts) with the same names as input .wav files"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Output directory to save generated .wav files"
    )
    parser.add_argument(
        "--remove_silence", action="store_true",
        help="Remove silence from generated audio"
    )
    parser.add_argument(
        "--vocab_file", type=str, required=True,
        help="Path to the vocabulary file (e.g., /path/to/vocab.txt)"
    )
    parser.add_argument(
        "--ckpt_file", type=str, required=True,
        help="Path to the model checkpoint file (e.g., /path/to/model.pt)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speed of the generated audio (0.3-2.0). Default is 1.0"
    )
    parser.add_argument(
        "--nfe_step", type=int, default=32,
        help="Number of NFE steps (16-64). Default is 32"
    )
    parser.add_argument(
        "--max_workers", type=int, default=None,
        help="Maximum number of parallel workers. Defaults to the number of available GPUs."
    )
    parser.add_argument(
        "--norm", type=str, required=False, default=None,
        help="Normalization type (e.g., 'hun', 'eng'). Determines which normalizer to use."
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="Random seed for reproducibility. Default is -1, which selects a random seed."
    )
    return parser.parse_args()

def process_files(
    wav_files,
    input_dir,
    input_gen_dir,
    output_dir,
    remove_silence,
    vocab_file,
    ckpt_file,
    speed,
    nfe_step,
    device,
    norm_value,
    seed,
):
    try:
        # Normalizáció inicializálása, ha szükséges
        if norm_value is not None:
            normaliser_path = PROJECT_ROOT / "normalisers" / norm_value / "normaliser.py"
            logger.info(f"Normaliser path: {normaliser_path}")
            if not normaliser_path.exists():
                logger.error(f"Normalizer module not found for norm='{norm_value}' at {normaliser_path}")
                sys.exit(1)
            spec = importlib.util.spec_from_file_location("normaliser", normaliser_path)
            normaliser = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(normaliser)
            if not hasattr(normaliser, 'normalize'):
                logger.error(f"The normalizer module '{normaliser_path}' does not have a 'normalize' function.")
                sys.exit(1)
            normalize_fn = normaliser.normalize
            logger.info(f"Loaded normalizer '{norm_value}' from {normaliser_path}")
        else:
            normalize_fn = None
            logger.info("No normalization will be applied as --norm parameter was not provided.")

        # F5TTS osztály inicializálása az adott device-ön
        f5tts = F5TTS(
            vocab_file=vocab_file,
            ckpt_file=ckpt_file,
            vocoder_name="vocos",
            device=device
        )
        logger.info(f"Initialized F5TTS on device {device}")

        for wav_path in tqdm.tqdm(wav_files, desc=f"Processing on {device}"):
            relative_path = wav_path.relative_to(input_dir)
            output_wav_path = output_dir / relative_path.parent / f"{wav_path.stem}.wav"
            output_wav_path.parent.mkdir(parents=True, exist_ok=True)

            if output_wav_path.exists():
                logger.info(f"Output file {output_wav_path} already exists. Skipping.")
                continue

            ref_txt_path = input_dir / relative_path.parent / f"{wav_path.stem}.txt"
            gen_txt_path = input_gen_dir / relative_path.parent / f"{wav_path.stem}.txt"

            if not ref_txt_path.exists():
                logger.warning(f"Reference text file not found for {wav_path.relative_to(input_dir)}, skipping.")
                continue

            if not gen_txt_path.exists():
                logger.warning(f"Generated text file not found for {wav_path.relative_to(input_gen_dir)}, skipping.")
                continue

            with open(ref_txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()

            with open(gen_txt_path, "r", encoding="utf-8") as f:
                gen_text = f.read().strip()
                if normalize_fn is not None:
                    try:
                        gen_text = normalize_fn(gen_text)
                        logger.debug(f"Normalized gen_text for {gen_txt_path}")
                    except Exception as e:
                        logger.error(f"Normalization failed for {gen_txt_path}: {e}")
                        continue

            try:
                f5tts.infer(
                    ref_file=str(wav_path),
                    ref_text=ref_text,
                    gen_text=gen_text,
                    file_wave=str(output_wav_path),
                    remove_silence=remove_silence,
                    speed=speed,
                    nfe_step=nfe_step,
                    seed=seed,
                )
                logger.info(f"Generated audio saved to {output_wav_path}")
            except Exception as e:
                logger.error(f"Error processing {wav_path.relative_to(input_dir)}: {e}", exc_info=True)
                continue

    except Exception as e:
        logger.critical(f"Critical error in process on device {device}: {e}", exc_info=True)

def main_worker(gpu, args, chunks, input_dir, input_gen_dir, output_dir):
    # Biztosítjuk, hogy a worker a megfelelő GPU-t használja
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    logger.info(f"Worker {gpu} using device {device}")

    process_files(
        chunks[gpu],
        input_dir,
        input_gen_dir,
        output_dir,
        args.remove_silence,
        args.vocab_file,
        args.ckpt_file,
        args.speed,
        args.nfe_step,
        device,
        args.norm,
        args.seed,
    )

def main():
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    input_gen_dir = Path(args.input_gen_dir)
    output_dir = Path(args.output_dir)

    if not (0.5 <= args.speed <= 2.0):
        logger.error(f"Invalid speed value: {args.speed}. Must be between 0.5 and 2.0.")
        sys.exit(1)
    if not (16 <= args.nfe_step <= 64):
        logger.error(f"Invalid nfe_step value: {args.nfe_step}. Must be between 16 and 64.")
        sys.exit(1)

    wav_files = list(input_dir.rglob("*.wav"))
    if not wav_files:
        logger.error(f"No .wav files found in {input_dir} or its subdirectories.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("No GPUs detected. Exiting.")
        sys.exit(1)
    logger.info(f"Number of available GPUs: {num_gpus}")

    # A fájlok szétosztása a GPU-k között
    chunks = [[] for _ in range(num_gpus)]
    for idx, wav_file in enumerate(wav_files):
        chunks[idx % num_gpus].append(wav_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    # A worker folyamatok indítása az mp.spawn segítségével
    mp.spawn(main_worker, nprocs=num_gpus, args=(args, chunks, input_dir, input_gen_dir, output_dir))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
