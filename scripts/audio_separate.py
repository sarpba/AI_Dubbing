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
import shutil  # Új import a fájlok másolásához

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
    # Save as WAV
    tensor = torch.from_numpy(audio_data)  # alak: (channels, length)
    torchaudio.save(path, tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)

def separate_audio_demucs(audio_path, output_dir, model, device):
    """
    Demucs segítségével szétválasztja az audio sávot beszédre és egyéb hangokra.
    Elmenti a szétválasztott hangfájlokat.
    """
    try:
        # Betöltjük a hangot
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 44100:
            print(f"Átalakítás 44100 Hz-re.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)
            waveform = resampler(waveform)
            sample_rate = 44100

        # waveform alakja: (channels, length)
        # Áthelyezzük az eszközre (CPU vagy GPU)
        waveform = waveform.to(device)

        # Alkalmazzuk a modellt
        with torch.no_grad():
            # Átalakítjuk az alakot: (batch_size, channels, length)
            estimates = apply_model(model, waveform.unsqueeze(0), device=device)

        # A források neveinek lekérése
        sources = model.sources  # Forrásnevek listája
        print(f"Modell forrásai: {sources}")

        # Ellenőrizzük, hogy a 'vocals' forrás létezik-e
        if 'vocals' not in sources:
            print("A 'vocals' forrás nem található a modell kimenetében.")
            return False

        # A 'vocals' indexének lekérése
        vocals_index = sources.index('vocals')

        # A 'vocals' kinyerése
        vocals = estimates[0, vocals_index].cpu().numpy()

        # A 'non_speech' kinyerése az összes többi forrás összegzésével
        other_indices = [i for i in range(len(sources)) if i != vocals_index]
        non_speech = estimates[0, other_indices].sum(dim=0).cpu().numpy()

        # Fájlok mentése
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, f"{base_name}_speech.wav")
        non_speech_path = os.path.join(output_dir, f"{base_name}_non_speech.wav")

        # Hangfájlok mentése torchaudio-val
        save_audio_torchaudio(vocals, vocals_path, sample_rate)
        save_audio_torchaudio(non_speech, non_speech_path, sample_rate)

        print(f"Szétválasztott fájlok mentve: {vocals_path}, {non_speech_path}")
        return True
    except Exception as e:
        print(f"Hiba történt a szétválasztás közben: {audio_path}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Videófájlokból audio kivonása és szétválasztása Demucs MDX segítségével.')
    parser.add_argument('-i', '--input_dir', required=True, help='Bemeneti könyvtár útvonala.')
    parser.add_argument('-o', '--output_dir', required=True, help='Kimeneti könyvtár útvonala.')
    parser.add_argument('--device', default='cuda', help='Eszköz: "cuda" vagy "cpu".')
    parser.add_argument('--keep_full_audio', action='store_true', help='Teljes audio fájl megtartása.')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    device = args.device if torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(input_dir):
        print(f"A megadott bemeneti könyvtár nem létezik: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Modell betöltése
    print("Demucs MDX modell betöltése...")
    try:
        model = get_model('mdx_extra')
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
                # Folytatjuk a szétválasztást még ha a mentés sikertelen is

        # Audio szétválasztása Demucs MDX modellel
        success = separate_audio_demucs(temp_audio_path, output_dir, model, device)
        if not success:
            # Ideiglenes fájl törlése, ha a szétválasztás sikertelen
            os.remove(temp_audio_path)
            continue

        # Ideiglenes fájl törlése, ha a teljes audio mentése nincs bekapcsolva
        if not args.keep_full_audio:
            os.remove(temp_audio_path)

    print("Feldolgozás kész.")

if __name__ == '__main__':
    main()

