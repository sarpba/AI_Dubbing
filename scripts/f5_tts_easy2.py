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
import re
import math

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

from cached_path import cached_path
try:
    import whisper
except ImportError:
    whisper = None
try:
    from transformers import pipeline
except ImportError:
    pipeline = None
from Levenshtein import distance as levenshtein_distance

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEL_SPEC_TYPE = "vocos"
NORMALIZER_TO_WHISPER_LANG = {"hun": "hu", "eng": "en"}
WHISPER_LANG_CODE_TO_NAME = {"hu": "hungarian", "en": "english"}


class F5TTS:
    # ... (Változatlan) ...
    def __init__(self, model_cls, model_cfg_dict, ckpt_file, vocab_file, vocoder_name=DEFAULT_MEL_SPEC_TYPE, ode_method="euler", use_ema=True, local_path=None, device=None):
        self.final_wave = None; self.target_sample_rate = target_sample_rate; self.hop_length = hop_length; self.seed = -1; self.mel_spec_type = vocoder_name; self.device = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.load_vocoder_model(vocoder_name, local_path); self.load_ema_model(model_cls, model_cfg_dict, ckpt_file, self.mel_spec_type, vocab_file, ode_method, use_ema)
    def load_vocoder_model(self, vocoder_name, local_path): self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)
    def load_ema_model(self, model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema): self.ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device)
    def export_wav(self, wav, file_wave, remove_silence=False): sf.write(file_wave, wav, self.target_sample_rate); (remove_silence_for_generated_wav(file_wave) if remove_silence else None)
    def infer(self, ref_file, ref_text, gen_text, file_wave, remove_silence=False, speed=1.0, nfe_step=32, seed=-1):
        if not (0.3 <= speed <= 2.0): raise ValueError(f"Invalid speed: {speed}")
        if not (16 <= nfe_step <= 64): raise ValueError(f"Invalid nfe_step: {nfe_step}")
        # A seed-et itt már nem kell véletlenszerűvé tenni, ha -1, mert a ciklus kezeli
        seed_everything(seed if seed != -1 else random.randint(0, sys.maxsize)); self.seed = seed; ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)
        wav, sr, spect = infer_process(ref_file, ref_text, gen_text, self.ema_model, self.vocoder, self.mel_spec_type, show_info=logger.info, progress=tqdm, target_rms=0.1, cross_fade_duration=0.15, nfe_step=nfe_step, cfg_strength=2, sway_sampling_coef=-1, speed=speed, fix_duration=None, device=self.device)
        if file_wave is not None: self.export_wav(wav, file_wave, remove_silence)
        return wav, sr, spect

def time_to_filename_str(seconds: float) -> str:
    # ... (Változatlan) ...
    if seconds < 0: seconds = 0
    td = datetime.timedelta(seconds=seconds)
    minutes, secs = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"

def resolve_model_paths_from_dir(model_dir_path: str):
    # ... (Változatlan) ...
    model_dir = Path(model_dir_path);
    if not model_dir.is_dir(): logger.error(f"Model directory not found: {model_dir}"); sys.exit(1)
    ckpt_files = list(model_dir.glob('*.pt')) + list(model_dir.glob('*.safetensors'))
    if not ckpt_files: logger.error(f"No checkpoint file found in {model_dir}"); sys.exit(1)
    if len(ckpt_files) > 1: logger.warning(f"Multiple checkpoints found, using {ckpt_files[0]}")
    resolved_ckpt_file = ckpt_files[0]; resolved_vocab_file = model_dir / 'vocab.txt'
    if not resolved_vocab_file.exists(): logger.error(f"vocab.txt not found in {model_dir}"); sys.exit(1)
    json_files = list(model_dir.glob('*.json'))
    if not json_files: logger.error(f"No config .json found in {model_dir}"); sys.exit(1)
    if len(json_files) > 1: logger.warning(f"Multiple configs found, using {json_files[0]}")
    resolved_json_file = json_files[0]
    logger.info(f"Found model files in '{model_dir}':\n  - Checkpoint: {resolved_ckpt_file.name}\n  - Vocab: {resolved_vocab_file.name}\n  - Config: {resolved_json_file.name}")
    return str(resolved_ckpt_file), str(resolved_vocab_file), str(resolved_json_file)

def interactive_model_selection(tts_base_path: Path):
    # ... (Változatlan) ...
    logger.info(f"No model directory specified. Searching for models in: {tts_base_path}")
    models = sorted([d for d in tts_base_path.iterdir() if d.is_dir()])
    if not models: logger.error(f"No models found in {tts_base_path}"); sys.exit(1)
    print("\nPlease select a model from the TTS directory:")
    for i, model_path in enumerate(models): print(f"  {i + 1}: {model_path.name}")
    while True:
        try:
            choice = int(input("Enter number: ")) - 1
            if 0 <= choice < len(models): selected_path = models[choice]; logger.info(f"Model selected: {selected_path.name}"); return str(selected_path)
            else: print("Invalid number, try again.")
        except ValueError: print("Invalid input, please enter a number.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="F5-TTS Inference Script with Dynamic, Conditional Whisper Verification.")
    parser.add_argument("project_name", type=str, help="A projekt könyvtárának neve a 'workdir' mappán belül.")
    parser.add_argument("--norm", type=str, required=True, help="Normalizálás típusa (pl. 'hun', 'eng'). Ezt használja a Whisper nyelv beállításához is.")
    parser.add_argument("--model_dir", type=str, default=None, help="Opcionális: a TTS modell könyvtárának elérési útja.")
    parser.add_argument("--speed", type=float, default=1.0, help="A generált hang sebessége (0.3-2.0). Alapértelmezett: 1.0.")
    parser.add_argument("--nfe_step", type=int, default=32, help="NFE lépések száma (16-64). Alapértelmezett: 32.")
    parser.add_argument("--max_workers", type=int, default=None, help="Párhuzamos workerek maximális száma. Alapértelmezetten az elérhető GPU-k száma.")
    parser.add_argument("--seed", type=int, default=-1, help="Véletlenszám-generátor magja. A '-1' érték véletlenszerűséget és a verifikációs folyamat aktiválását jelenti. Alapértelmezett: -1.")
    parser.add_argument("--remove_silence", action="store_true", help="Csend eltávolítása a generált hangból.")
    # === MÓDOSÍTOTT VERIFIKÁCIÓS ARGUMENTUMOK ===
    parser.add_argument("--max-retries", type=int, default=5, help="Maximális újragenerálási kísérletek száma (csak random seed esetén). Alapértelmezett: 5.")
    parser.add_argument("--tolerance-factor", type=float, default=1.0, help="Tolerancia szorzó a szavak száma alapján (csak random seed esetén). Alapértelmezett: 1.0.")
    parser.add_argument("--min-tolerance", type=int, default=2, help="A dinamikusan számított tolerancia minimális értéke (csak random seed esetén). Alapértelmezett: 2.")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-large-v3", help="A verifikációhoz használt Whisper modell. Lehet OpenAI név ('large-v3') vagy Hugging Face azonosító ('user/model'). Alapértelmezett: openai/whisper-large-v3.")
    parser.add_argument("--beam-size", type=int, default=5, help="A Whisper dekódoláshoz használt nyalábszélesség (csak random seed esetén). Alapértelmezett: 5.")
    args = parser.parse_args()
    return args

# === VISSZAÁLLÍTOTT NORMALIZÁLÓ FÜGGVÉNY ===
def normalize_text_for_comparison(text: str) -> str:
    """
    Prepares text for comparison by lowercasing, removing punctuation, and collapsing whitespace.
    "Hello,   world!" -> "hello world"
    """
    text = text.lower().strip()
    text = re.sub(r'[.,?!-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def main_worker(gpu, args, all_chunks, input_wav_path_str, stats):
    tasks_chunk = all_chunks[gpu]
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    logger.info(f"Worker {gpu} starting on device {device} with {len(tasks_chunk)} tasks.")

    try:
        # === A VERIFIKÁCIÓS ESZKÖZÖK BETÖLTÉSE CSAK SZÜKSÉG ESETÉN ===
        transcriber = None
        run_transcription = None
        if args.seed == -1:
            logger.info(f"Worker {gpu}: Random seed detected. Initializing verification tools.")
            whisper_language = NORMALIZER_TO_WHISPER_LANG.get(args.norm.lower())
            if whisper_language is None:
                logger.error(f"Worker {gpu}: Unsupported language mapping for '{args.norm}'."); return

            is_huggingface_model = '/' in args.whisper_model
            if is_huggingface_model:
                if pipeline is None:
                    logger.error("Hugging Face model requested, but `transformers` library is not installed."); return
                logger.info(f"Worker {gpu}: Loading Hugging Face model '{args.whisper_model}'...")
                transcriber = pipeline("automatic-speech-recognition", model=args.whisper_model, torch_dtype=torch.float16, device=device)
            else:
                if whisper is None:
                    logger.error("OpenAI model requested, but `openai-whisper` library is not installed."); return
                model_name_for_openai = args.whisper_model.replace("openai/", "")
                logger.info(f"Worker {gpu}: Loading OpenAI Whisper model '{model_name_for_openai}'...")
                transcriber = whisper.load_model(model_name_for_openai, device=device)
            
            logger.info(f"Worker {gpu}: Whisper model loaded. Using beam size: {args.beam_size}.")

            def create_transcription_func(transcriber_obj, is_hf, lang_code, beam_size):
                def func(audio_path):
                    if is_hf:
                        lang_name = WHISPER_LANG_CODE_TO_NAME.get(lang_code)
                        generate_kwargs = {"language": lang_name, "num_beams": beam_size} if lang_name else {"num_beams": beam_size}
                        if not lang_name: logger.warning(f"No full language name mapping for '{lang_code}'.")
                        return transcriber_obj(audio_path, generate_kwargs=generate_kwargs)
                    else:
                        return transcriber_obj.transcribe(audio_path, language=lang_code, fp16=torch.cuda.is_available(), beam_size=beam_size)
                return func
            
            run_transcription = create_transcription_func(transcriber, is_huggingface_model, whisper_language, args.beam_size)
        
        # --- F5-TTS modell betöltése (mindig szükséges) ---
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

        architecture_class_map={"DiT":DiT,"UNetT":UNetT};resolved_model_cls=architecture_class_map.get(args.resolved_model_architecture)
        if resolved_model_cls is None:logger.error(f"Worker {gpu}: Unknown model architecture '{args.resolved_model_architecture}'.");sys.exit(1)
        f5tts=F5TTS(model_cls=resolved_model_cls,model_cfg_dict=args.resolved_model_params_dict,ckpt_file=args.resolved_ckpt_file,vocab_file=args.resolved_vocab_file,device=device)
        logger.info(f"Worker {gpu}: Initialized '{args.resolved_model_architecture}' model on device {device}.")
        full_audio_data,sample_rate=sf.read(input_wav_path_str);logger.info(f"Worker {gpu}: Loaded main audio file {Path(input_wav_path_str).name}")

        for segment in tqdm.tqdm(tasks_chunk, desc=f"Processing on {device}", position=gpu):
            start_time = segment.get('start'); end_time = segment.get('end')
            ref_text = segment.get('text', '').strip(); gen_text = segment.get('translated_text', '').strip()
            if not all([isinstance(start_time,(int,float)),isinstance(end_time,(int,float)),ref_text,gen_text]): continue
            
            filename = f"{time_to_filename_str(start_time)}_{time_to_filename_str(end_time)}.wav"
            output_wav_path = Path(args.output_dir) / filename
            if output_wav_path.exists(): stats['successful'] += 1; continue
            
            if normalize_fn:
                try: gen_text = normalize_fn(gen_text)
                except Exception: stats['failed'] += 1; continue

            start_sample=int(start_time*sample_rate);end_sample=int(end_time*sample_rate)
            if end_sample>len(full_audio_data):end_sample=len(full_audio_data)
            if start_sample>=end_sample:logger.warning(f"Worker {gpu}: Skipping invalid time range {start_time}-{end_time}."); continue
            ref_audio_chunk=full_audio_data[start_sample:end_sample]

            # === FELTÉTELES ELLENŐRZÉSI LOGIKA ===
            if args.seed == -1:
                # --- VERIFIKÁCIÓS ÚTVONAL (VÉLETLEN SEED) ---
                verification_passed = False
                for attempt in range(1, args.max_retries + 1):
                    temp_gen_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False,dir=args.output_dir) as tmp_gen_file:temp_gen_path=tmp_gen_file.name
                        with tempfile.NamedTemporaryFile(suffix=".wav",delete=True) as tmp_ref_file:
                            sf.write(tmp_ref_file.name,ref_audio_chunk,sample_rate)
                            f5tts.infer(ref_file=tmp_ref_file.name,ref_text=ref_text,gen_text=gen_text,file_wave=temp_gen_path,remove_silence=args.remove_silence,speed=args.speed,nfe_step=args.nfe_step,seed=-1)
                        
                        transcription_result = run_transcription(temp_gen_path)
                        transcribed_text = transcription_result['text']

                        norm_original = normalize_text_for_comparison(gen_text)
                        norm_transcribed = normalize_text_for_comparison(transcribed_text)
                        dist = levenshtein_distance(norm_original, norm_transcribed)
                        word_count = len(gen_text.split()); calculated_tolerance = math.ceil(word_count*args.tolerance_factor); final_tolerance = max(calculated_tolerance, args.min_tolerance)
                        
                        if dist <= final_tolerance:
                            logger.info(f"Worker {gpu}: Verification PASSED (Attempt {attempt}/{args.max_retries}, Dist: {dist}, Allowed: {final_tolerance})")
                            os.rename(temp_gen_path, output_wav_path); verification_passed = True; break
                        else:
                            logger.warning(f"Worker {gpu}: Verification FAILED (Attempt {attempt}/{args.max_retries}, Dist: {dist}, Allowed: {final_tolerance})\n  - Expected: '{norm_original}'\n  - Got:      '{norm_transcribed}'")
                            os.remove(temp_gen_path)
                    except Exception as e:
                        logger.error(f"Worker {gpu}: Error on attempt {attempt} for segment {start_time}-{end_time}: {e}", exc_info=False)
                        if temp_gen_path and os.path.exists(temp_gen_path): os.remove(temp_gen_path)
                        continue

                if verification_passed: stats['successful'] += 1
                else: logger.error(f"Worker {gpu}: All {args.max_retries} attempts FAILED for segment {start_time}-{end_time}."); stats['failed'] += 1
            else:
                # --- KÖZVETLEN GENERÁLÁSI ÚTVONAL (FIX SEED) ---
                logger.info(f"Worker {gpu}: Fixed seed ({args.seed}) - Generating segment {start_time}-{end_time} without verification.")
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_ref_file:
                        sf.write(tmp_ref_file.name, ref_audio_chunk, sample_rate)
                        f5tts.infer(ref_file=tmp_ref_file.name,ref_text=ref_text,gen_text=gen_text,file_wave=str(output_wav_path),remove_silence=args.remove_silence,speed=args.speed,nfe_step=args.nfe_step,seed=args.seed)
                    stats['successful'] += 1
                except Exception as e:
                    logger.error(f"Worker {gpu}: Direct generation FAILED for segment {start_time}-{end_time} with fixed seed: {e}", exc_info=False)
                    stats['failed'] += 1

    except Exception as e:
        logger.critical(f"Critical error in worker {gpu}: {e}", exc_info=True)

def process_file_pair(args, input_wav_path, input_json_path):
    # ... (Változatlan) ...
    logger.info(f"Starting processing for WAV: '{input_wav_path.name}' and JSON: '{input_json_path.name}'")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        try: data = json.load(f); segments = data.get("segments", [])
        except json.JSONDecodeError as e: logger.error(f"JSON decode error in {input_json_path.name}: {e}"); return
    if not segments: logger.error(f"No 'segments' in {input_json_path.name}"); return
    segments.sort(key=lambda x: x.get('start', 0))
    try: full_audio_data, sample_rate = sf.read(input_wav_path); duration_seconds = len(full_audio_data) / sample_rate
    except Exception as e: logger.error(f"Failed to load audio {input_wav_path.name}: {e}"); return
    logger.info(f"Audio loaded. Duration: {duration_seconds:.2f}s. Rate: {sample_rate}Hz.")
    logger.info("Extracting non-segmented (noise) parts...")
    last_end_time = 0.0; output_dir_noise_path = Path(args.output_dir_noise)
    for segment in segments:
        start_time = segment.get('start')
        if start_time is None: continue
        if start_time > last_end_time:
            noise_output_path = output_dir_noise_path / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(start_time)}.wav"
            if not noise_output_path.exists(): sf.write(noise_output_path, full_audio_data[int(last_end_time*sample_rate):int(start_time*sample_rate)], sample_rate)
        last_end_time = segment.get('end', start_time)
    if duration_seconds > last_end_time:
        noise_output_path = output_dir_noise_path / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(duration_seconds)}.wav"
        if not noise_output_path.exists(): sf.write(noise_output_path, full_audio_data[int(last_end_time*sample_rate):], sample_rate)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: logger.error("No GPUs detected. Exiting."); sys.exit(1)
    max_workers = args.max_workers if args.max_workers is not None else num_gpus
    num_workers = min(num_gpus, max_workers)
    if num_workers <= 0: logger.error("Number of workers must be positive. Exiting."); sys.exit(1)
    logger.info(f"Using {num_workers} GPU(s) for TTS generation.")
    chunks = [[] for _ in range(num_workers)]; [chunks[idx % num_workers].append(segment) for idx, segment in enumerate(segments)]
    with mp.Manager() as manager:
        stats = manager.dict({'successful': 0, 'failed': 0, 'total': len(segments)})
        logger.info("Starting TTS generation workers...")
        spawn_args = (args, chunks, str(input_wav_path), stats)
        mp.spawn(main_worker, nprocs=num_workers, args=spawn_args)
        logger.info(f"Finished TTS generation for {input_wav_path.name}.")
        successful_count = stats['successful']; failed_count = stats['failed']; total_count = stats['total']
        processed_count = successful_count + failed_count
        logger.info("="*50 + "\nFINAL STATISTICS" + f"\n  - Total Segments to Process: {total_count}\n  - Processed Segments: {processed_count}\n  - Successfully Generated & Verified: {successful_count}\n  - Failed (after all retries): {failed_count}\n" + "="*50)

def main():
    # ... (Változatlan) ...
    config_path = PROJECT_ROOT / 'config.json'
    if not config_path.exists(): logger.error(f"Config file not found: {config_path}"); sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
    cfg_dirs = config_data['DIRECTORIES']; cfg_subdirs = config_data['PROJECT_SUBDIRS']
    args = parse_arguments()
    workdir_path = PROJECT_ROOT / cfg_dirs['workdir']; full_project_path = workdir_path / args.project_name
    if not full_project_path.is_dir(): logger.error(f"Project directory not found: {full_project_path}"); sys.exit(1)
    input_wav_dir = full_project_path / cfg_subdirs['separated_audio_speech']; input_json_dir = full_project_path / cfg_subdirs['translated']
    output_dir = full_project_path / cfg_subdirs['translated_splits']; output_dir_noise = full_project_path / cfg_subdirs['noice_splits']
    output_dir.mkdir(parents=True, exist_ok=True); output_dir_noise.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir); args.output_dir_noise = str(output_dir_noise); args.normalisers_dir = str(PROJECT_ROOT / cfg_dirs['normalisers'])
    if args.model_dir: model_dir_path = args.model_dir
    else: tts_base_path = PROJECT_ROOT / cfg_dirs['TTS']; model_dir_path = interactive_model_selection(tts_base_path)
    ckpt_path, vocab_path, json_path = resolve_model_paths_from_dir(model_dir_path)
    args.resolved_ckpt_file = ckpt_path; args.resolved_vocab_file = vocab_path
    with open(json_path, 'r', encoding='utf-8') as f: model_config_data = json.load(f)
    if "model_architecture" not in model_config_data or "model_params" not in model_config_data: logger.error(f"Model config '{json_path}' is missing keys."); sys.exit(1)
    args.resolved_model_architecture = model_config_data["model_architecture"]; args.resolved_model_params_dict = model_config_data["model_params"]
    wav_files = list(input_wav_dir.glob('*.wav')); json_files = list(input_json_dir.glob('*.json'))
    if not wav_files: logger.error(f"No .wav files in {input_wav_dir}"); sys.exit(1)
    if len(wav_files) > 1: logger.warning(f"Multiple .wav files in {input_wav_dir}, using {wav_files[0]}")
    if not json_files: logger.error(f"No .json files in {input_json_dir}"); sys.exit(1)
    if len(json_files) > 1: logger.warning(f"Multiple .json files in {input_json_dir}, using {json_files[0]}")
    input_wav_path = wav_files[0]; input_json_path = json_files[0]
    process_file_pair(args, input_wav_path, input_json_path)
    logger.info("Script finished.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()