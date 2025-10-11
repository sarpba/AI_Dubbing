#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import torch
import json
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import numpy as np
import traceback
import shutil

def extract_audio(video_path, audio_path):
    """
    Kivonja az audio sávot a videófájlból sztereó formátumban (44.1 kHz).
    """
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-ac', '2', '-ar', '44100', '-vn', audio_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Hiba történt az audio kivonása közben: {video_path}")
        print(result.stderr.decode())
        return False
    if not os.path.exists(audio_path):
        print(f"Az audio fájl nem jött létre: {audio_path}")
        return False
    if os.path.getsize(audio_path) == 0:
        print(f"Az audio fájl üres: {audio_path}")
        return False
    return True

def save_audio_torchaudio(audio_data, path, sample_rate=44100):
    """
    Mentés torchaudio segítségével WAV formátumban 16-bit PCM.
    """
    # Ha a bemenet mono, itt nem kell duplikálni, a torchaudio.save kezeli
    # A lényeg, hogy a Demucs kimenete már sztereó lesz
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=0)
        
    # Normalizálás és konvertálás a mentéshez
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    audio_data = audio_data.astype(np.float32)
    tensor = torch.from_numpy(audio_data)
    
    # Ha a kimeneti tenzor mégis mono lenne, itt is biztosítjuk a sztereót
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(2, 1)

    torchaudio.save(path, tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)


def separate_audio_demucs(audio_path, base_name, speech_output_dir, background_output_dir, model, device, non_speech_silence=False):
    """
    Demucs segítségével szétválasztja az audio sávot (egész fájlon).
    Elmenti a szétválasztott hangfájlokat a megfelelő könyvtárakba.
    Ha a non_speech_silence kapcsoló be van kapcsolva, akkor a non_speech részt csenddé alakítja.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # --- JAVÍTÁS: Mono bemenet kezelése ---
        if waveform.shape[0] == 1:
            print("Mono bemenet észlelve, átalakítás sztereóvá a Demucs feldolgozáshoz.")
            waveform = waveform.repeat(2, 1)
        # --- JAVÍTÁS VÉGE ---

        if sample_rate != 44100:
            print("Átalakítás 44100 Hz-re.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)
            waveform = resampler(waveform)
            sample_rate = 44100

        waveform = waveform.to(device)

        with torch.no_grad():
            estimates = apply_model(model, waveform.unsqueeze(0), device=device)

        sources = model.sources
        print(f"Modell forrásai: {sources}")

        if 'vocals' not in sources:
            print("A 'vocals' forrás nem található a modell kimenetében.")
            return False

        vocals_index = sources.index('vocals')
        vocals = estimates[0, vocals_index].cpu().numpy()
        other_indices = [i for i in range(len(sources)) if i != vocals_index]
        non_speech = estimates[0, other_indices].sum(dim=0).cpu().numpy()

        if non_speech_silence:
            non_speech = np.zeros_like(non_speech)

        vocals_path = os.path.join(speech_output_dir, f"{base_name}_speech.wav")
        non_speech_path = os.path.join(background_output_dir, f"{base_name}_non_speech.wav")

        save_audio_torchaudio(vocals, vocals_path, sample_rate)
        save_audio_torchaudio(non_speech, non_speech_path, sample_rate)

        print(f"Szétválasztott fájlok mentve: {vocals_path}, {non_speech_path}")
        return True
    except Exception as e:
        print(f"Hiba történt a szétválasztás közben: {audio_path}")
        traceback.print_exc()
        return False

def separate_audio_demucs_chunks(audio_path, base_name, speech_output_dir, background_output_dir, model, device, chunk_size_min, non_speech_silence=False):
    """
    Az audiófilet chunkokra bontja, minden darabon lefuttatja a szétválasztást,
    végül összefűzi és elmenti a végső speech és non_speech fájlokat a megfelelő könyvtárakba.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # --- JAVÍTÁS: Mono bemenet kezelése ---
        if waveform.shape[0] == 1:
            print("Mono bemenet észlelve, átalakítás sztereóvá a Demucs feldolgozáshoz.")
            waveform = waveform.repeat(2, 1)
        # --- JAVÍTÁS VÉGE ---

        if sample_rate != 44100:
            print("Átalakítás 44100 Hz-re (chunk alapú feldolgozás).")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)
            waveform = resampler(waveform)
            sample_rate = 44100

        chunk_size_samples = int(chunk_size_min * 60 * sample_rate)
        total_samples = waveform.shape[1]
        vocals_chunks = []
        non_speech_chunks = []

        print(f"Chunk méret: {chunk_size_samples} mintát, összesen {total_samples} minta.")

        for start in range(0, total_samples, chunk_size_samples):
            end = min(total_samples, start + chunk_size_samples)
            chunk = waveform[:, start:end].to(device)
            
            # A chunk-nak is sztereónak kell lennie, a fenti átalakítás ezt biztosítja
            if chunk.shape[0] != 2:
                 print(f"Hiba: A feldolgozandó chunk nem sztereó! Méret: {chunk.shape}")
                 # Biztonsági ellenőrzés, bár elvileg nem futhatna ide
                 if chunk.shape[0] == 1:
                     chunk = chunk.repeat(2, 1)
                 else: # Ha több mint 2 csatorna lenne valamiért
                     chunk = chunk[:2, :]
            
            with torch.no_grad():
                estimates = apply_model(model, chunk.unsqueeze(0), device=device)

            sources = model.sources
            if 'vocals' not in sources:
                print("A 'vocals' forrás nem található a modell kimenetében.")
                return False
            vocals_index = sources.index('vocals')
            vocals_chunk = estimates[0, vocals_index].cpu().numpy()
            other_indices = [i for i in range(len(sources)) if i != vocals_index]
            non_speech_chunk = estimates[0, other_indices].sum(dim=0).cpu().numpy()
            if non_speech_silence:
                non_speech_chunk = np.zeros_like(non_speech_chunk)
            vocals_chunks.append(vocals_chunk)
            non_speech_chunks.append(non_speech_chunk)
            print(f"Chunk {start}-{end} feldolgozva.")

        vocals_combined = np.concatenate(vocals_chunks, axis=1)
        non_speech_combined = np.concatenate(non_speech_chunks, axis=1)

        vocals_path = os.path.join(speech_output_dir, f"{base_name}_speech.wav")
        non_speech_path = os.path.join(background_output_dir, f"{base_name}_non_speech.wav")

        save_audio_torchaudio(vocals_combined, vocals_path, sample_rate)
        save_audio_torchaudio(non_speech_combined, non_speech_path, sample_rate)

        print(f"Szétválasztott (összefűzött) fájlok mentve: {vocals_path}, {non_speech_path}")
        return True
    except Exception as e:
        print(f"Hiba történt a chunk alapú szétválasztás közben: {audio_path}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Audiófájlok szétválasztása Demucs/MDX segítségével egy projekt könyvtárstruktúráján belül.'
    )
    parser.add_argument('-p', '--project', required=True, help='A projekt neve a "workdir" könyvtáron belül.')
    parser.add_argument('--device', default='cuda', help='Eszköz: "cuda" vagy "cpu".')
    parser.add_argument('--keep_full_audio', action='store_true', help='A konvertált teljes audio fájl megtartása a "separated_audio_speech" mappában.')
    parser.add_argument(
        '--non_speech_silence',
        action='store_true',
        help='Ha aktiválva, a non_speech fájlban csak csend lesz.'
    )
    parser.add_argument(
        '--chunk_size',
        type=float,
        default=5,
        help='Darabolás hossza percben (alapértelmezett: 5 perc). 0 érték esetén a teljes fájl egyszerre lesz feldolgozva.'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=["htdemucs", "htdemucs_ft", "hdemucs_mmi", "mdx", "mdx_extra"],
        default="htdemucs",
        help='A használt modell kiválasztása (alapértelmezett: htdemucs).'
    )
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        workdir_name = config['DIRECTORIES']['workdir']
        project_dir = os.path.join(project_root, workdir_name, args.project)

        if not os.path.isdir(project_dir):
            print(f"Hiba: A projekt könyvtár nem létezik: {project_dir}")
            sys.exit(1)

        input_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['extracted_audio'])
        speech_output_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        background_output_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_background'])

    except FileNotFoundError:
        print("Hiba: A config.json nem található. A szkriptnek a megfelelő könyvtárszerkezetben kell lennie.")
        sys.exit(1)
    except KeyError as e:
        print(f"Hiba: A config.json feldolgozása közben. Hiányzó kulcs: {e}")
        sys.exit(1)

    if not os.path.isdir(input_dir):
        print(f"Hiba: A bemeneti könyvtár nem létezik: {input_dir}")
        sys.exit(1)

    os.makedirs(speech_output_dir, exist_ok=True)
    os.makedirs(background_output_dir, exist_ok=True)

    print(f"Demucs/MDX modell betöltése: {args.model}...")
    try:
        model = get_model(args.model)
        model.to(device)
        model.eval()
        print("Modell betöltve.")
    except Exception as e:
        print(f"Hiba a modell betöltése közben: {e}")
        sys.exit(1)

    # --- ÚJ LOGIKA: Fájlok csoportosítása és a center sáv priorizálása ---
    files_by_group = {}
    for f in os.listdir(input_dir):
        if not f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            continue

        base = os.path.splitext(f)[0]
        # Potenciális többcsatornás suffixek eltávolítása a csoportnév azonosításához
        # Azért, hogy pl. 'audio_FC' és 'audio_FL' egy csoportba kerüljön ('audio')
        group_name = base
        # Gyakori csatorna jelölések
        suffixes = ["_FC", "_FL", "_FR", "_SL", "_SR", "_RL", "_RR", "_LFE", "_C", "_L", "_R"]
        for suffix in suffixes:
            if base.upper().endswith(suffix):
                group_name = base[:-len(suffix)]
                break
        
        if group_name not in files_by_group:
            files_by_group[group_name] = []
        files_by_group[group_name].append(f)

    # --- Módosított ciklus a csoportosított fájlok feldolgozására ---
    for group_name, file_list in files_by_group.items():
        filename_to_process = None
        
        # Keresés a '_FC.wav' végződésű fájlra a csoporton belül
        for f in file_list:
            if os.path.splitext(f)[0].upper().endswith("_FC"):
                filename_to_process = f
                print(f"\nCenter sáv (_FC) kiválasztva a '{group_name}' csoporthoz: {f}")
                break
        
        # Ha nincs '_FC.wav', akkor a lista első elemét választjuk
        if not filename_to_process:
            filename_to_process = file_list[0]
            if len(file_list) > 1:
                 print(f"\nNem található '_FC.wav' a '{group_name}' csoportban. Automatikus választás: {filename_to_process}")

        audio_path = os.path.join(input_dir, filename_to_process)
        base_name = os.path.splitext(filename_to_process)[0]
        temp_audio_path = os.path.join(speech_output_dir, f"{base_name}_temp.wav")

        print(f"--- Feldolgozás: {audio_path} ---")

        if filename_to_process.lower().endswith('.wav'):
            temp_audio_path = audio_path
            was_converted = False
        else:
            print(f"Konvertálás WAV formátumba: {temp_audio_path}")
            if not extract_audio(audio_path, temp_audio_path):
                continue
            was_converted = True

        if args.keep_full_audio:
            full_audio_path = os.path.join(speech_output_dir, f"{base_name}_full.wav")
            try:
                shutil.copy(temp_audio_path, full_audio_path)
                print(f"Teljes audio mentve: {full_audio_path}")
            except Exception as e:
                print(f"Hiba a teljes audio mentése közben: {e}")

        if args.chunk_size > 0:
            success = separate_audio_demucs_chunks(
                temp_audio_path, base_name,
                speech_output_dir, background_output_dir,
                model, device,
                chunk_size_min=args.chunk_size,
                non_speech_silence=args.non_speech_silence
            )
        else:
            success = separate_audio_demucs(
                temp_audio_path, base_name,
                speech_output_dir, background_output_dir,
                model, device,
                non_speech_silence=args.non_speech_silence
            )

        if was_converted and os.path.exists(temp_audio_path) and temp_audio_path != audio_path:
            os.remove(temp_audio_path)
        
        if not success:
            print(f"A szétválasztás sikertelen volt a következő fájlnál: {filename_to_process}")

    print("\n--- Feldolgozás befejezve ---")
    print(f"A beszéd sávok a következő könyvtárba kerültek: {speech_output_dir}")
    print(f"A háttérzaj sávok a következő könyvtárba kerültek: {background_output_dir}")

if __name__ == '__main__':
    main()