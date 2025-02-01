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
import shutil
import logging

# Naplózás konfigurálása
logging.basicConfig(
    level=logging.DEBUG,  # Részletes naplózás
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("separate.log"),  # Naplófájl
        logging.StreamHandler(sys.stdout)    # Kimenet a konzolra
    ]
)

logger = logging.getLogger(__name__)

def extract_audio(video_path, audio_path):
    """
    Kivonja az audio sávot a videófájlból sztereó formátumban (44.1 kHz).
    """
    logger.info(f"Audio kivonása: {video_path} -> {audio_path}")
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-ac', '2', '-ar', '44100', '-vn', audio_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error(f"Hiba történt az audio kivonása közben: {video_path}")
        logger.error(result.stderr.decode())
        return False
    # Ellenőrizzük, hogy a fájl létezik és nem üres
    if not os.path.exists(audio_path):
        logger.error(f"Az audio fájl nem jött létre: {audio_path}")
        return False
    if os.path.getsize(audio_path) == 0:
        logger.error(f"Az audio fájl üres: {audio_path}")
        return False
    logger.debug(f"Audio sikeresen kivonva: {audio_path}")
    return True

def save_audio_torchaudio(audio_data, path, sample_rate=44100):
    """
    Mentés torchaudio segítségével WAV formátumban 16-bit PCM.
    """
    logger.debug(f"Audio mentése: {path}")
    # Normalizáljuk a hangot
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    # Convert to float32
    audio_data = audio_data.astype(np.float32)
    # Save as WAV
    tensor = torch.from_numpy(audio_data)  # alak: (channels, length)
    torchaudio.save(path, tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)
    logger.debug(f"Audio mentve: {path}")

def separate_audio_demucs(audio_path, output_dir, model, device):
    """
    Demucs segítségével szétválasztja az audio sávot beszédre és egyéb hangokra.
    Elmenti a szétválasztott hangfájlokat.
    """
    logger.info(f"Audio szétválasztása Demucs-szal: {audio_path}")
    try:
        # Betöltjük a hangot
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.debug(f"Hangkimenet betöltve: {audio_path}, mintavételi frekvencia: {sample_rate}")
        if sample_rate != 44100:
            logger.info(f"Mintavételi frekvencia átalakítása 44100 Hz-re.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)
            waveform = resampler(waveform)
            sample_rate = 44100

        # Áthelyezzük az eszközre (CPU vagy GPU)
        waveform = waveform.to(device)
        logger.debug(f"Waveform áthelyezve a(z) {device} eszközre.")

        # Alkalmazzuk a modellt
        with torch.no_grad():
            # Átalakítjuk az alakot: (batch_size, channels, length)
            estimates = apply_model(model, waveform.unsqueeze(0), device=device)
        logger.debug("Modell alkalmazva az audio hullámra.")

        # A források neveinek lekérése
        sources = model.sources  # Forrásnevek listája
        logger.info(f"Modell forrásai: {sources}")

        # Ellenőrizzük, hogy a 'vocals' forrás létezik-e
        if 'vocals' not in sources:
            logger.error("A 'vocals' forrás nem található a modell kimenetében.")
            return False

        # A 'vocals' indexének lekérése
        vocals_index = sources.index('vocals')

        # A 'vocals' kinyerése
        vocals = estimates[0, vocals_index].cpu().numpy()
        logger.debug("Vokálok kinyerve a modellből.")

        # A 'non_speech' kinyerése az összes többi forrás összegzésével
        other_indices = [i for i in range(len(sources)) if i != vocals_index]
        non_speech = estimates[0, other_indices].sum(dim=0).cpu().numpy()
        logger.debug("Nem beszéd hangok kinyerve és összegezve.")

        # Fájlok mentése
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, f"{base_name}_speech.wav")
        non_speech_path = os.path.join(output_dir, f"{base_name}_non_speech.wav")

        # Hangfájlok mentése torchaudio-val
        save_audio_torchaudio(vocals, vocals_path, sample_rate)
        save_audio_torchaudio(non_speech, non_speech_path, sample_rate)

        logger.info(f"Szétválasztott fájlok mentve: {vocals_path}, {non_speech_path}")
        return True
    except Exception as e:
        logger.error(f"Hiba történt a szétválasztás közben: {audio_path}")
        logger.error(traceback.format_exc())
        return False

def split_audio(audio_path, temp_dir, segment_time=300):
    """
    Felosztja az audio fájlt adott időtartamú szegmensekre (alapértelmezés szerint 5 perc).
    """
    logger.info(f"Audio felosztása {segment_time/60} perces szegmensekre: {audio_path}")
    os.makedirs(temp_dir, exist_ok=True)
    split_pattern = os.path.join(temp_dir, "segment_%03d.wav")
    command = [
        'ffmpeg', '-y', '-i', audio_path,
        '-f', 'segment',
        '-segment_time', str(segment_time),
        '-c', 'copy',
        split_pattern
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error("Hiba történt az audio felosztása közben.")
        logger.error(result.stderr.decode())
        return False
    # Ellenőrizzük, hogy létrejöttek-e a szegmensek
    segments = sorted([f for f in os.listdir(temp_dir) if f.startswith("segment_") and f.endswith(".wav")])
    if not segments:
        logger.error("Nincsenek létrehozott szegmensek.")
        return False
    logger.debug(f"{len(segments)} szegmens létrehozva.")
    return True

def concatenate_audio(processed_dir, output_vocals_path, output_non_speech_path):
    """
    Összeilleszti az összes szétválasztott 'non_speech' és 'speech' fájlt a végleges kimeneti fájlokká.
    Először a '_non_speech.wav' fájlokat, majd a '_speech.wav' fájlokat.
    """
    logger.info(f"Nem beszéd hangok összeillesztése: {processed_dir}")

    # Összegyűjtjük és sorba rendezzük a 'non_speech' fájlokat
    non_speech_files = sorted([f for f in os.listdir(processed_dir) if f.endswith("_non_speech.wav")])

    # Nem beszéd hangok összeillesztése
    if non_speech_files:
        concatenated_non_speech = []
        for nf in non_speech_files:
            path = os.path.join(processed_dir, nf)
            waveform, sr = torchaudio.load(path)
            concatenated_non_speech.append(waveform)
        if concatenated_non_speech:
            non_speech_tensor = torch.cat(concatenated_non_speech, dim=1)
            torchaudio.save(output_non_speech_path, non_speech_tensor, sr, encoding="PCM_S", bits_per_sample=16)
            logger.info(f"Nem beszéd hangok összeillesztve: {output_non_speech_path}")

        # Egyedi '_non_speech.wav' fájlok törlése
        for nf in non_speech_files:
            path = os.path.join(processed_dir, nf)
            try:
                os.remove(path)
                logger.debug(f"Fájl törölve: {path}")
            except Exception as e:
                logger.error(f"Hiba a fájl törlése közben: {path}")
                logger.error(str(e))
    else:
        logger.warning("Nincsenek 'non_speech' szegmensek az összeillesztéshez.")

    logger.info(f"Vokálok összeillesztése: {processed_dir}")

    # Összegyűjtjük és sorba rendezzük a 'speech' fájlokat
    vocals_files = sorted([f for f in os.listdir(processed_dir) if f.endswith("_speech.wav")])

    # Vokálok összeillesztése
    if vocals_files:
        concatenated_vocals = []
        for vf in vocals_files:
            path = os.path.join(processed_dir, vf)
            waveform, sr = torchaudio.load(path)
            concatenated_vocals.append(waveform)
        if concatenated_vocals:
            vocals_tensor = torch.cat(concatenated_vocals, dim=1)
            torchaudio.save(output_vocals_path, vocals_tensor, sr, encoding="PCM_S", bits_per_sample=16)
            logger.info(f"Vokálok összeillesztve: {output_vocals_path}")
    else:
        logger.warning("Nincsenek 'vocals' szegmensek az összeillesztéshez.")

def main():
    parser = argparse.ArgumentParser(description='Videófájlokból audio kivonása és szétválasztása Demucs segítségével.')
    parser.add_argument('-i', '--input_dir', required=True, help='Bemeneti könyvtár útvonala.')
    parser.add_argument('-o', '--output_dir', required=True, help='Kimeneti könyvtár útvonala.')
    parser.add_argument('--device', default='cuda', help='Eszköz: "cuda" vagy "cpu".')
    parser.add_argument('--keep_full_audio', action='store_true', help='Teljes audio fájl megtartása.')
    parser.add_argument('--model', default='htdemucs', help='Demucs model to use')  # Új argumentum

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Használt eszköz: {device}")

    if not os.path.isdir(input_dir):
        logger.error(f"A megadott bemeneti könyvtár nem létezik: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Kimeneti könyvtár létrehozva vagy már létezik: {output_dir}")

    # Modell betöltése
    logger.info(f"Demucs modell betöltése: {args.model}")
    try:
        model = get_model(args.model)
        logger.debug("Modell letöltve.")
    except Exception as e:
        logger.error("Hiba a modell betöltése közben.")
        logger.error(str(e))
        sys.exit(1)

    model.to(device)
    model.eval()
    logger.info("Modell betöltve és eszközre helyezve.")

    # Iterálunk a bemeneti könyvtárban található fájlokon
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            logger.debug(f"Fájl kihagyva, nem videó: {filename}")
            continue

        video_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        temp_audio_path = os.path.join(output_dir, f"{base_name}_temp.wav")

        logger.info(f"Feldolgozás: {video_path}")

        # Teljes audio kivonása sztereóban
        success = extract_audio(video_path, temp_audio_path)
        if not success:
            logger.warning(f"Audio kivonása sikertelen: {video_path}")
            continue

        # Ha a kapcsoló be van kapcsolva, mentsük el a teljes audiofájlt
        if args.keep_full_audio:
            full_audio_path = os.path.join(output_dir, f"{base_name}_full.wav")
            try:
                shutil.copy(temp_audio_path, full_audio_path)
                logger.info(f"Teljes audio mentve: {full_audio_path}")
            except Exception as e:
                logger.error(f"Hiba a teljes audio mentése közben: {full_audio_path}")
                logger.error(str(e))
                # Folytatjuk a szétválasztást még ha a mentés sikertelen is

        # Ideiglenes könyvtár létrehozása a szegmenseknek
        temp_dir = os.path.join(output_dir, "temp", base_name)
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Időközi könyvtár létrehozva: {temp_dir}")

        # Audio felosztása 5 perces szegmensekre
        success = split_audio(temp_audio_path, temp_dir)
        if not success:
            logger.warning(f"Audio felosztása sikertelen: {temp_audio_path}")
            # Ideiglenes fájl törlése
            try:
                os.remove(temp_audio_path)
                logger.debug(f"Ideiglenes fájl törölve: {temp_audio_path}")
            except Exception as e:
                logger.error(f"Hiba az ideiglenes fájl törlése közben: {temp_audio_path}")
                logger.error(str(e))
            continue

        # Feldolgozott szegmensek könyvtára
        processed_dir = os.path.join(temp_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        logger.debug(f"Feldolgozott szegmensek könyvtára létrehozva: {processed_dir}")

        # Feldolgozzuk az egyes szegmenseket
        segments = sorted([f for f in os.listdir(temp_dir) if f.startswith("segment_") and f.endswith(".wav")])
        for segment_file in segments:
            segment_path = os.path.join(temp_dir, segment_file)
            logger.info(f"Szegmens feldolgozása: {segment_path}")
            success = separate_audio_demucs(segment_path, processed_dir, model, device)
            if not success:
                logger.warning(f"Szétválasztás sikertelen a szegmensben: {segment_path}")
                continue

        # Végleges fájlok elérési útjának meghatározása
        final_vocals_path = os.path.join(output_dir, f"{base_name}_speech.wav")
        final_non_speech_path = os.path.join(output_dir, f"{base_name}_non_speech.wav")

        # Összeillesztjük a szétválasztott szegmenseket
        concatenate_audio(processed_dir, final_vocals_path, final_non_speech_path)

        # Ideiglenes könyvtár törlése
        try:
            shutil.rmtree(os.path.join(output_dir, "temp"))
            logger.debug(f"Időközi könyvtár törölve: {os.path.join(output_dir, 'temp')}")
        except Exception as e:
            logger.error(f"Hiba az időközi könyvtár törlése közben: {os.path.join(output_dir, 'temp')}")
            logger.error(str(e))

        # Ideiglenes teljes audiofájl törlése, ha nem kell megőrizni
        if not args.keep_full_audio:
            try:
                os.remove(temp_audio_path)
                logger.debug(f"Ideiglenes fájl törölve: {temp_audio_path}")
            except Exception as e:
                logger.error(f"Hiba az ideiglenes fájl törlése közben: {temp_audio_path}")
                logger.error(str(e))

    logger.info("Feldolgozás kész.")

if __name__ == '__main__':
    main()
