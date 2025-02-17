#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import numpy as np
import traceback
import shutil  # Fájlok másolásához

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
    # Ellenőrizzük, hogy a fájl létezik és nem üres
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
    # Normalizáljuk a hangot
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    # Convert to float32
    audio_data = audio_data.astype(np.float32)
    # Alakítsuk tensorra: (channels, length)
    tensor = torch.from_numpy(audio_data)
    torchaudio.save(path, tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)

def separate_audio_demucs(audio_path, output_dir, model, device, non_speech_silence=False):
    """
    Demucs segítségével szétválasztja az audio sávot (egész fájlon).
    Elmenti a szétválasztott hangfájlokat.
    Ha a non_speech_silence kapcsoló be van kapcsolva, akkor a non_speech részt csenddé alakítja.
    """
    try:
        # Betöltjük a hangot
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 44100:
            print("Átalakítás 44100 Hz-re.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)
            waveform = resampler(waveform)
            sample_rate = 44100

        # Áthelyezzük az eszközre (CPU vagy GPU)
        waveform = waveform.to(device)

        # Alkalmazzuk a modellt (teljes audióra)
        with torch.no_grad():
            estimates = apply_model(model, waveform.unsqueeze(0), device=device)

        sources = model.sources  # Források listája
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

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, f"{base_name}_speech.wav")
        non_speech_path = os.path.join(output_dir, f"{base_name}_non_speech.wav")

        save_audio_torchaudio(vocals, vocals_path, sample_rate)
        save_audio_torchaudio(non_speech, non_speech_path, sample_rate)

        print(f"Szétválasztott fájlok mentve: {vocals_path}, {non_speech_path}")
        return True
    except Exception as e:
        print(f"Hiba történt a szétválasztás közben: {audio_path}")
        traceback.print_exc()
        return False

def separate_audio_demucs_chunks(audio_path, output_dir, base_name, model, device, chunk_size_min, non_speech_silence=False):
    """
    Az audiófilet chunkokra bontja (chunk mérete percekben megadva), majd minden darabon lefuttatja a szétválasztást,
    végül összefűzi az eredményeket, és elmenti a végső speech és non_speech fájlokat.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
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
            chunk = waveform[:, start:end]
            chunk = chunk.to(device)
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

        # Az egyes chunkok összefűzése a time tengely mentén (második dimenzió)
        vocals_combined = np.concatenate(vocals_chunks, axis=1)
        non_speech_combined = np.concatenate(non_speech_chunks, axis=1)

        vocals_path = os.path.join(output_dir, f"{base_name}_speech.wav")
        non_speech_path = os.path.join(output_dir, f"{base_name}_non_speech.wav")

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
        description='Videófájlokból audio kivonása és szétválasztása Demucs/MDX segítségével.'
    )
    parser.add_argument('-i', '--input_dir', required=True, help='Bemeneti könyvtár útvonala.')
    parser.add_argument('-o', '--output_dir', required=True, help='Kimeneti könyvtár útvonala.')
    parser.add_argument('--device', default='cuda', help='Eszköz: "cuda" vagy "cpu".')
    parser.add_argument('--keep_full_audio', action='store_true', help='Teljes audio fájl megtartása.')
    parser.add_argument(
        '--non_speech_silence',
        action='store_true',
        help='Ha aktiválva, a non_speech fájlban csak csend lesz.'
    )
    parser.add_argument(
        '--chunk_size',
        type=float,
        default=5,
        help='Darabolás hossza percben (alapértelmezett: 5 perc).'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=["htdemucs", "htdemucs_ft", "hdemucs_mmi", "mdx", "mdx_extra"],
        default="htdemucs",
        help='A használt modell kiválasztása.'
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    device = args.device if torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(input_dir):
        print(f"A megadott bemeneti könyvtár nem létezik: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Modell betöltése a felhasználó által kiválasztott modell alapján
    print(f"Demucs/MDX modell betöltése: {args.model}...")
    try:
        model = get_model(args.model)
    except Exception as e:
        print("Hiba a modell betöltése közben.")
        print(str(e))
        sys.exit(1)

    model.to(device)
    model.eval()
    print("Modell betöltve.")

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            continue

        video_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        temp_audio_path = os.path.join(output_dir, f"{base_name}_temp.wav")

        print(f"Feldolgozás: {video_path}")

        # Teljes audio kivonása sztereóban
        success = extract_audio(video_path, temp_audio_path)
        if not success:
            continue

        # Ha a kapcsoló be van kapcsolva, mentsük el a teljes audiofájlt
        if args.keep_full_audio:
            full_audio_path = os.path.join(output_dir, f"{base_name}_full.wav")
            try:
                shutil.copy(temp_audio_path, full_audio_path)
                print(f"Teljes audio mentve: {full_audio_path}")
            except Exception as e:
                print(f"Hiba a teljes audio mentése közben: {full_audio_path}")
                print(str(e))
                # Folytatjuk a szétválasztást, még ha a mentés sikertelen is

        # Chunk alapú feldolgozás, ha chunk_size > 0
        if args.chunk_size > 0:
            success = separate_audio_demucs_chunks(
                temp_audio_path, output_dir, base_name,
                model, device,
                chunk_size_min=args.chunk_size,
                non_speech_silence=args.non_speech_silence
            )
        else:
            # Ha nincs chunkolás, a teljes fájlra futtatjuk a szétválasztást
            success = separate_audio_demucs(
                temp_audio_path, output_dir, model, device,
                non_speech_silence=args.non_speech_silence
            )
        if not success:
            os.remove(temp_audio_path)
            continue

        # Ideiglenes fájl törlése, ha a teljes audio mentése nincs bekapcsolva
        if not args.keep_full_audio:
            os.remove(temp_audio_path)

    print("Feldolgozás kész.")

if __name__ == '__main__':
    main()
