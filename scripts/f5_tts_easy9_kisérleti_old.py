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
import numpy as np

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
try:
    from num2words import num2words
except ImportError:
    num2words = None
from Levenshtein import distance as levenshtein_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(processName)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("infer_batch.log")])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEL_SPEC_TYPE = "vocos"
NORMALIZER_TO_WHISPER_LANG = {"hun": "hu", "eng": "en"}
WHISPER_LANG_CODE_TO_NAME = {"hu": "hungarian", "en": "english"}


class F5TTS:
    def __init__(self, model_cls, model_cfg_dict, ckpt_file, vocab_file, vocoder_name=DEFAULT_MEL_SPEC_TYPE, ode_method="euler", use_ema=True, local_path=None, device=None):
        self.final_wave=None; self.target_sample_rate=target_sample_rate; self.hop_length=hop_length; self.seed=-1; self.mel_spec_type=vocoder_name; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu"); self.load_vocoder_model(vocoder_name, local_path); self.load_ema_model(model_cls, model_cfg_dict, ckpt_file, self.mel_spec_type, vocab_file, ode_method, use_ema)
    def load_vocoder_model(self, vocoder_name, local_path): self.vocoder=load_vocoder(vocoder_name, local_path is not None, local_path, self.device)
    def load_ema_model(self, model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema): self.ema_model=load_model(model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device)
    def export_wav(self, wav, file_wave, remove_silence=False): sf.write(file_wave, wav, self.target_sample_rate); (remove_silence_for_generated_wav(file_wave) if remove_silence else None)
    def infer(self, ref_file, ref_text, gen_text, file_wave, remove_silence=False, speed=1.0, nfe_step=32, seed=-1):
        if not (0.3 <= speed <= 2.0): raise ValueError(f"Invalid speed: {speed}")
        if not (16 <= nfe_step <= 64): raise ValueError(f"Invalid nfe_step: {nfe_step}")
        current_seed=seed if seed != -1 else random.randint(0, sys.maxsize)
        seed_everything(current_seed); self.seed=current_seed; ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)
        wav, sr, spect = infer_process(ref_file, ref_text, gen_text, self.ema_model, self.vocoder, self.mel_spec_type, show_info=logger.info, progress=tqdm, target_rms=0.1, cross_fade_duration=0.15, nfe_step=nfe_step, cfg_strength=2, sway_sampling_coef=-1, speed=speed, fix_duration=None, device=self.device)
        if file_wave is not None: self.export_wav(wav, file_wave, remove_silence)
        return wav, sr, spect

def time_to_filename_str(seconds: float) -> str:
    if seconds < 0: seconds=0
    td=datetime.timedelta(seconds=seconds); minutes, secs=divmod(td.seconds, 60); hours, minutes=divmod(minutes, 60); milliseconds=td.microseconds // 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"

def resolve_model_paths_from_dir(model_dir_path: str):
    model_dir=Path(model_dir_path);
    if not model_dir.is_dir(): logger.error(f"Model directory not found: {model_dir}"); sys.exit(1)
    ckpt_files=list(model_dir.glob('*.pt')) + list(model_dir.glob('*.safetensors'))
    if not ckpt_files: logger.error(f"No checkpoint file found in {model_dir}"); sys.exit(1)
    if len(ckpt_files) > 1: logger.warning(f"Multiple checkpoints found, using {ckpt_files[0]}")
    resolved_ckpt_file=ckpt_files[0]; resolved_vocab_file=model_dir/'vocab.txt'
    if not resolved_vocab_file.exists(): logger.error(f"vocab.txt not found in {model_dir}"); sys.exit(1)
    json_files=list(model_dir.glob('*.json'))
    if not json_files: logger.error(f"No config .json found in {model_dir}"); sys.exit(1)
    if len(json_files) > 1: logger.warning(f"Multiple configs found, using {json_files[0]}")
    resolved_json_file=json_files[0]
    logger.info(f"Found model files in '{model_dir}':\n  - Checkpoint: {resolved_ckpt_file.name}\n  - Vocab: {resolved_vocab_file.name}\n  - Config: {resolved_json_file.name}")
    return str(resolved_ckpt_file), str(resolved_vocab_file), str(resolved_json_file)

def interactive_model_selection(tts_base_path: Path):
    logger.info(f"No model directory specified. Searching for models in: {tts_base_path}")
    models=sorted([d for d in tts_base_path.iterdir() if d.is_dir()])
    if not models: logger.error(f"No models found in {tts_base_path}"); sys.exit(1)
    print("\nPlease select a model from the TTS directory:")
    for i, model_path in enumerate(models): print(f"  {i + 1}: {model_path.name}")
    while True:
        try:
            choice=int(input("Enter number: ")) - 1
            if 0 <= choice < len(models): selected_path=models[choice]; logger.info(f"Model selected: {selected_path.name}"); return str(selected_path)
            else: print("Invalid number, try again.")
        except ValueError: print("Invalid input, please enter a number.")
        
def parse_arguments():
    parser = argparse.ArgumentParser(description="F5-TTS Inference Script with advanced verification and debugging.")
    parser.add_argument("project_name", type=str, help="A projekt könyvtárának neve a 'workdir' mappán belül.")
    parser.add_argument("--norm", type=str, required=True, help="Normalizálás típusa (pl. 'hun', 'eng').")
    parser.add_argument("--model_dir", type=str, default=None, help="Opcionális: a TTS modell könyvtárának elérési útja.")
    parser.add_argument("--speed", type=float, default=1.0, help="A generált hang sebessége (0.3-2.0).")
    parser.add_argument("--nfe_step", type=int, default=32, help="NFE lépések száma (16-64).")
    parser.add_argument("--remove_silence", action="store_true", help="Csend eltávolítása a generált hangból.")
    parser.add_argument("--normalize-ref-audio", action="store_true", help="Aktiválja a referencia audio csúcshangerejének normalizálását.")
    parser.add_argument("--ref-audio-peak", type=float, default=0.95, help="A normalizált referencia audio cél csúcsértéke (0.0-1.0).")
    parser.add_argument("--max_workers", type=int, default=None, help="Párhuzamos workerek maximális száma.")
    parser.add_argument("--seed", type=int, default=-1, help="Véletlenszám-generátor magja. A '-1' véletlenszerűséget és a verifikációt jelenti.")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximális újragenerálási kísérletek száma (csak random seed esetén).")
    parser.add_argument("--tolerance-factor", type=float, default=1.0, help="Tolerancia szorzó a szavak száma alapján (csak random seed esetén).")
    parser.add_argument("--min-tolerance", type=int, default=2, help="A dinamikusan számított tolerancia minimális értéke (csak random seed esetén).")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-large-v3", help="A verifikációhoz használt Whisper modell.")
    parser.add_argument("--beam-size", type=int, default=5, help="A Whisper dekódoláshoz használt nyalábszélesség (csak random seed esetén).")
    parser.add_argument("--save-failures", action="store_true", help="Elmenti a hibás generálásokat egy 'failed_generations' mappába.")
    parser.add_argument("--double-ref-on-failure", action="store_true", help="Az első hibás generálási kísérlet után megduplázza a referencia audiót és szöveget.")
    args = parser.parse_args()
    return args

def normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    if not 0.0 < target_peak <= 1.0: target_peak = 0.95
    current_peak = np.max(np.abs(audio))
    if current_peak == 0: return audio
    return audio * (target_peak / current_peak)

def normalize_text_for_comparison(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[.,?!-]', '', text); text = re.sub(r'\s+', ' ', text)
    return text

def convert_numbers_to_words(text: str, lang: str) -> str:
    if num2words is None: return text
    def replace_match(match):
        try: return num2words(int(match.group(0)), lang=lang)
        except Exception: return match.group(0)
    return re.sub(r'\b\d+\b', replace_match, text)


def main_worker(gpu, args, all_chunks, input_wav_path_str, stats, failed_segments_info):
    tasks_chunk = all_chunks[gpu]
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    logger.info(f"Worker {gpu} starting on device {device} with {len(tasks_chunk)} tasks.")

    try:
        run_transcription = None; whisper_language = None
        if args.seed == -1:
            if num2words is None: logger.error("Verification requires 'num2words'."); return
            whisper_language = NORMALIZER_TO_WHISPER_LANG.get(args.norm.lower())
            is_huggingface_model = '/' in args.whisper_model
            transcriber = pipeline("automatic-speech-recognition", model=args.whisper_model, torch_dtype=torch.float16, device=device) if is_huggingface_model else whisper.load_model(args.whisper_model.replace("openai/",""), device=device)
            def create_transcription_func(t, is_hf, lang, beam):
                def func(p):
                    if is_hf: k={"language":WHISPER_LANG_CODE_TO_NAME.get(lang),"num_beams":beam}; return t(p,generate_kwargs=k)
                    else: return t.transcribe(p,language=lang,fp16=torch.cuda.is_available(),beam_size=beam)
                return func
            run_transcription = create_transcription_func(transcriber, is_huggingface_model, whisper_language, args.beam_size)
        
        normalize_fn=None; normaliser_path=Path(args.normalisers_dir)/args.norm/"normaliser.py"
        if normaliser_path.exists():
            try:
                spec=importlib.util.spec_from_file_location("normaliser",normaliser_path); normaliser=importlib.util.module_from_spec(spec); spec.loader.exec_module(normaliser)
                if hasattr(normaliser, 'normalize'): normalize_fn=normaliser.normalize
            except Exception as e: logger.error(f"Worker {gpu}: Failed to load normalizer: {e}.")
        
        architecture_class_map={"DiT":DiT,"UNetT":UNetT}; resolved_model_cls=architecture_class_map.get(args.resolved_model_architecture)
        f5tts=F5TTS(model_cls=resolved_model_cls,model_cfg_dict=args.resolved_model_params_dict,ckpt_file=args.resolved_ckpt_file,vocab_file=args.resolved_vocab_file,device=device)
        
        full_audio_data, sample_rate = sf.read(input_wav_path_str)

        for segment in tqdm.tqdm(tasks_chunk, desc=f"Processing on {device}", position=gpu):
            start_time=segment.get('start'); end_time=segment.get('end')
            original_ref_text=segment.get('text','').strip(); original_gen_text=segment.get('translated_text','').strip()
            if not all([isinstance(start_time,(int,float)),isinstance(end_time,(int,float)),original_ref_text,original_gen_text]): continue
            
            filename=f"{time_to_filename_str(start_time)}_{time_to_filename_str(end_time)}.wav"; output_wav_path=Path(args.output_dir)/filename
            if output_wav_path.exists(): stats['successful']+=1; continue
            
            gen_text = original_gen_text
            if normalize_fn:
                try: gen_text=normalize_fn(gen_text)
                except Exception: stats['failed']+=1; failed_segments_info.append({'filename': filename, 'text': original_gen_text}); continue

            start_sample=int(start_time*sample_rate); end_sample=int(end_time*sample_rate)
            if end_sample>len(full_audio_data): end_sample=len(full_audio_data)
            if start_sample>=end_sample: continue
            
            original_ref_chunk = full_audio_data[start_sample:end_sample]
            processed_ref_chunk = original_ref_chunk.copy()
            if args.normalize_ref_audio:
                processed_ref_chunk = normalize_peak(processed_ref_chunk, args.ref_audio_peak)

            current_ref_text=original_ref_text
            current_ref_audio_chunk=processed_ref_chunk

            if args.seed == -1:
                verification_passed=False
                for attempt in range(1, args.max_retries+1):
                    temp_gen_path=None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False,dir=args.output_dir) as tmp: temp_gen_path=tmp.name
                        with tempfile.NamedTemporaryFile(suffix=".wav",delete=True) as tmp_ref:
                            sf.write(tmp_ref.name, current_ref_audio_chunk, sample_rate)
                            f5tts.infer(ref_file=tmp_ref.name, ref_text=current_ref_text, gen_text=gen_text, file_wave=temp_gen_path, seed=-1, speed=args.speed, nfe_step=args.nfe_step)
                        
                        raw_transcribed_text=run_transcription(temp_gen_path)['text']; converted_transcribed_text=convert_numbers_to_words(raw_transcribed_text,lang=whisper_language)
                        norm_original=normalize_text_for_comparison(gen_text); norm_transcribed=normalize_text_for_comparison(converted_transcribed_text)
                        dist=levenshtein_distance(norm_original,norm_transcribed)

                        # --- START OF MODIFICATION: Add penalty for extra initial 'a' or 'e' ---
                        original_words = norm_original.split()
                        transcribed_words = norm_transcribed.split()

                        if original_words and transcribed_words: # Ensure lists are not empty
                            # Condition: transcribed starts with 'a' or 'e', AND it's not the same as the original's first word
                            if transcribed_words[0] in ('a', 'e') and transcribed_words[0] != original_words[0]:
                                # Further check: the rest of the transcribed text should match the start of the original text.
                                # This ensures we're penalizing an *added* word, not a simple substitution.
                                rest_of_transcribed = ' '.join(transcribed_words[1:])
                                if norm_original.startswith(rest_of_transcribed):
                                    logger.warning(f"Worker {gpu}: Extra initial sound '{transcribed_words[0]}' detected. Applying penalty of 5. Original dist: {dist}.")
                                    dist += 5
                        # --- END OF MODIFICATION ---
                        
                        word_count=len(gen_text.split()); calculated_tolerance=math.ceil(word_count*args.tolerance_factor); final_tolerance=max(calculated_tolerance,args.min_tolerance)
                        
                        if dist <= final_tolerance:
                            os.rename(temp_gen_path, output_wav_path); verification_passed=True; break
                        else:
                            logger.warning(f"Worker {gpu}: Verification FAILED (Attempt {attempt}/{args.max_retries}) -> Distance: {dist}, Tolerance: {final_tolerance} for '{gen_text}'")
                            
                            if args.save_failures:
                                debug_segment_dir=Path(args.failed_generations_dir)/output_wav_path.stem; debug_segment_dir.mkdir(exist_ok=True)
                                failed_attempt_filename = filename.replace('.wav', f'_attempt_{attempt}.wav')
                                debug_gen_audio_path = debug_segment_dir / failed_attempt_filename
                                os.rename(temp_gen_path, debug_gen_audio_path)
                                
                                info_json_path=debug_segment_dir/"info.json"
                                info={}
                                if attempt==1:
                                    sf.write(debug_segment_dir/"ref_audio_original.wav", original_ref_chunk, sample_rate)
                                    if args.normalize_ref_audio:
                                        sf.write(debug_segment_dir/"ref_audio_normalized.wav", processed_ref_chunk, sample_rate)
                                    info={"ref_text":original_ref_text,"gen_text":gen_text,"failures":[]}
                                elif info_json_path.exists():
                                    with open(info_json_path,'r',encoding='utf-8') as f: info=json.load(f)
                                info.get("failures",[]).append({'attempt':attempt,'raw_transcribed_text':raw_transcribed_text,'converted_transcribed_text':converted_transcribed_text,'distance':dist,'allowed_tolerance':final_tolerance})
                                with open(info_json_path,'w',encoding='utf-8') as f: json.dump(info,f,ensure_ascii=False,indent=2)
                            else:
                                os.remove(temp_gen_path)

                            if args.double_ref_on_failure and attempt == 1:
                                logger.info(f"Worker {gpu}: Doubling reference for subsequent retries.")
                                current_ref_text=f"{original_ref_text} {original_ref_text}"
                                current_ref_audio_chunk=np.concatenate([processed_ref_chunk, processed_ref_chunk])
                                if args.save_failures:
                                    sf.write(debug_segment_dir/"ref_audio_doubled_normalized.wav", current_ref_audio_chunk, sample_rate)
                    
                    except Exception as e:
                        logger.error(f"Worker {gpu}: Error during verification attempt {attempt} for '{filename}': {e}", exc_info=True)
                        if temp_gen_path and os.path.exists(temp_gen_path): os.remove(temp_gen_path)
                        continue # Continue to the next attempt

                if verification_passed: stats['successful']+=1
                else: stats['failed']+=1; failed_segments_info.append({'filename': filename, 'text': original_gen_text})

            else: # Fixed seed, no verification loop
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_ref:
                        sf.write(tmp_ref.name, current_ref_audio_chunk, sample_rate)
                        f5tts.infer(ref_file=tmp_ref.name,ref_text=current_ref_text,gen_text=gen_text,file_wave=str(output_wav_path),seed=args.seed,speed=args.speed,nfe_step=args.nfe_step)
                    stats['successful']+=1
                except Exception as e: 
                    logger.error(f"Worker {gpu}: Failed to generate with fixed seed for '{filename}': {e}")
                    stats['failed']+=1
                    failed_segments_info.append({'filename': filename, 'text': original_gen_text})

    except Exception as e: logger.critical(f"Critical error in worker {gpu}: {e}", exc_info=True)

def process_file_pair(args, input_wav_path, input_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f: data=json.load(f); segments=data.get("segments", [])
    segments.sort(key=lambda x: x.get('start', 0))
    full_audio_data, sample_rate=sf.read(input_wav_path); duration_seconds=len(full_audio_data)/sample_rate
    last_end_time=0.0; output_dir_noise_path=Path(args.output_dir_noise)
    for segment in segments:
        start_time=segment.get('start');
        if start_time is not None and start_time > last_end_time:
            noise_output_path=output_dir_noise_path/f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(start_time)}.wav"
            if not noise_output_path.exists(): sf.write(noise_output_path, full_audio_data[int(last_end_time*sample_rate):int(start_time*sample_rate)], sample_rate)
        last_end_time=segment.get('end', start_time)
    if duration_seconds > last_end_time:
        noise_output_path=output_dir_noise_path/f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(duration_seconds)}.wav"
        if not noise_output_path.exists(): sf.write(noise_output_path, full_audio_data[int(last_end_time*sample_rate):], sample_rate)
    
    num_gpus=torch.cuda.device_count(); max_workers=args.max_workers if args.max_workers is not None else num_gpus; num_workers=min(num_gpus,max_workers)
    if num_workers == 0: logger.error("No GPUs available for processing. Exiting."); sys.exit(1)
    chunks=[[] for _ in range(num_workers)]; [chunks[idx%num_workers].append(s) for idx,s in enumerate(segments)]
    
    with mp.Manager() as manager:
        stats=manager.dict({'successful':0,'failed':0,'total':len(segments)})
        failed_segments_info = manager.list()
        
        spawn_args=(args, chunks, str(input_wav_path), stats, failed_segments_info)
        mp.spawn(main_worker, nprocs=num_workers, args=spawn_args)
        
        s=stats['successful']; f=stats['failed']; t=stats['total']; p=s+f
        logger.info("="*50 + f"\nFINAL STATISTICS\n  - Total: {t}\n  - Processed: {p}\n  - Successful: {s}\n  - Failed: {f}\n" + "="*50)
        
        if failed_segments_info:
            logger.info("\n" + "="*50)
            logger.info("FAILED SEGMENTS REPORT")
            sorted_failures = sorted(list(failed_segments_info), key=lambda x: x['filename'])
            for failure in sorted_failures:
                logger.info(f"  - {failure['filename']}: {failure['text']}")
            logger.info("="*50)

def main():
    config_path=PROJECT_ROOT/'config.json';
    with open(config_path,'r',encoding='utf-8') as f: config_data=json.load(f)
    cfg_dirs=config_data['DIRECTORIES']; cfg_subdirs=config_data['PROJECT_SUBDIRS']
    args=parse_arguments()
    full_project_path = (PROJECT_ROOT / cfg_dirs['workdir']) / args.project_name
    
    args.output_dir = str(full_project_path / cfg_subdirs['translated_splits'])
    args.output_dir_noise = str(full_project_path / cfg_subdirs['noice_splits'])
    args.normalisers_dir = str(PROJECT_ROOT / cfg_dirs['normalisers'])
    args.failed_generations_dir = str(full_project_path / "failed_generations")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir_noise).mkdir(parents=True, exist_ok=True)
    if args.save_failures and args.seed == -1: Path(args.failed_generations_dir).mkdir(parents=True, exist_ok=True)

    if args.model_dir: model_dir_path = args.model_dir
    else: tts_base_path=PROJECT_ROOT/cfg_dirs['TTS']; model_dir_path=interactive_model_selection(tts_base_path)
    
    ckpt_path, vocab_path, json_path = resolve_model_paths_from_dir(model_dir_path)
    args.resolved_ckpt_file=ckpt_path; args.resolved_vocab_file=vocab_path
    with open(json_path, 'r', encoding='utf-8') as f: model_config_data=json.load(f)
    args.resolved_model_architecture=model_config_data["model_architecture"]; args.resolved_model_params_dict=model_config_data["model_params"]
    
    input_wav_dir=full_project_path/cfg_subdirs['separated_audio_speech']; input_json_dir=full_project_path/cfg_subdirs['translated']
    wav_files=list(input_wav_dir.glob('*.wav')); json_files=list(input_json_dir.glob('*.json'))
    
    if not wav_files or not json_files:
        logger.error(f"Could not find required .wav or .json files in project '{args.project_name}'")
        logger.error(f"Searched in: {input_wav_dir} and {input_json_dir}")
        sys.exit(1)

    input_wav_path=wav_files[0]; input_json_path=json_files[0]
    process_file_pair(args, input_wav_path, input_json_path)
    logger.info("Script finished.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()