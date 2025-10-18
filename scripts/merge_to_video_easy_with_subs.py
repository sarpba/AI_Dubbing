import argparse
import subprocess
import json
import os
import sys
import glob
import re
from pathlib import Path

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode


def find_first_video_file(directory):
    """Megkeresi az első videófájlt a megadott könyvtárban."""
    supported_formats = ('.mkv', '.mp4', '.avi', '.mov', '.webm')
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(supported_formats):
            return os.path.join(directory, f)
    return None


def find_latest_audio_file(directory):
    """Megkeresi a legfrissebb .wav fájlt a megadott könyvtárban."""
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    if not wav_files:
        return None
    return max(wav_files, key=os.path.getmtime)


def detect_language_code(srt_path):
    """Megpróbálja kinyerni a kétbetűs nyelvi kódot a fájlnévből."""
    base = os.path.basename(srt_path)
    match = re.search(r'_([a-zA-Z]{2})\.srt$', base)
    if match:
        return match.group(1).lower()
    return None


def collect_srt_files(directory):
    """Összegyűjti a könyvtárban található SRT feliratfájlokat nyelvi kóddal együtt."""
    srt_files = []
    for path in sorted(glob.glob(os.path.join(directory, '*.srt'))):
        lang = detect_language_code(path)
        srt_files.append((path, lang))
    return srt_files


def get_target_sample_rate(video_file):
    """Lekéri a video első audió streamjének mintavételi rátáját."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'json',
            video_file,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        return info['streams'][0].get('sample_rate', '48000')
    except Exception:
        return '48000'


def process_audio_to_raw_aac(audio_wav_file, target_sample_rate):
    """A bemeneti WAV fájlt átkódolja nyers .aac (ADTS) formátumra."""
    processed_audio_path = "processed_audio_temp.aac"
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_wav_file,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', str(target_sample_rate),
            '-ac', '2',
            processed_audio_path,
        ]
        print("\n--- FFMPEG LOG START (Creating raw .aac) ---")
        subprocess.run(cmd, check=True)
        print("--- FFMPEG LOG END (Creating raw .aac) ---\n")
        return processed_audio_path
    except subprocess.CalledProcessError:
        print("Hiba az audió .aac-re való feldolgozása során. Lásd a fenti ffmpeg logot.")
        sys.exit(1)


def build_ffmpeg_command(video_file, new_audio_aac_file, srt_entries, language, target_sample_rate, output_file):
    """Felépíti az ffmpeg parancsot a végső multiplexeléshez."""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-i', new_audio_aac_file,
    ]

    for srt, _ in srt_entries:
        cmd.extend(['-i', srt])

    cmd.extend([
        '-map', '0:v:0',
        '-map', '0:a:0',
        '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a:0', 'copy',
        '-c:a:1', 'aac',
        '-ar:a:1', str(target_sample_rate),
        '-b:a:1', '128k',
        '-metadata:s:a:1', f'language={language}',
    ])

    subtitle_start_index = 2  # video=0, audio=1, így a felirat bemenetek 2-től indulnak
    subtitle_output_index = 0
    for idx, (_, lang_code) in enumerate(srt_entries):
        input_index = subtitle_start_index + idx
        cmd.extend([
            '-map', f'{input_index}:s:0',
        ])
        cmd.extend([
            f'-c:s:{subtitle_output_index}', 'copy',
        ])
        if lang_code:
            cmd.extend([
                f'-metadata:s:s:{subtitle_output_index}', f'language={lang_code}',
            ])
        subtitle_output_index += 1

    cmd.append(output_file)
    return cmd


def add_audio_and_subs_to_video(video_file, new_audio_aac_file, srt_entries, language, target_sample_rate, output_dir):
    """Hozzáadja az új audiósávot és a külön SRT fájlokat a videó konténeréhez."""
    video_basename = os.path.basename(video_file)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = os.path.join(output_dir, f"{video_name}_with_{language}_dub{video_ext}")

    cmd = build_ffmpeg_command(video_file, new_audio_aac_file, srt_entries, language, target_sample_rate, output_file)

    print(f"FFmpeg parancs futtatása:\n{' '.join(cmd)}")
    print("\n--- FFMPEG LOG START (Final Muxing) ---")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Hiba az audiósáv vagy feliratok hozzáadása során. Lásd a fenti ffmpeg logot.")
        sys.exit(1)
    finally:
        print("--- FFMPEG LOG END (Final Muxing) ---\n")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Videóhoz új, szinkronizált audiosávot fűz és az upload könyvtár SRT feliratait is hozzáadja.")
    parser.add_argument('project_name', help='A feldolgozandó projekt könyvtár neve a "workdir"-en belül.')
    parser.add_argument('-lang', '--language', required=True, help='A hozzáadandó audiosáv nyelvi címkéje (pl. hun, eng).')
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)

    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Hiba a config.json betöltése közben: {e}")
        sys.exit(1)

    try:
        workdir_base = config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        project_path = os.path.join(workdir_base, args.project_name)
        upload_dir = os.path.join(project_path, subdirs['upload'])
        audio_dir = os.path.join(project_path, subdirs['film_dubbing'])
        output_dir = os.path.join(project_path, subdirs['download'])
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        sys.exit(1)

    input_video = find_first_video_file(upload_dir)
    if not input_video:
        print(f"Hiba: Nem található videófájl a(z) '{upload_dir}' könyvtárban.")
        sys.exit(1)
    print(f"Bemeneti videó: {input_video}")

    input_wav_audio = find_latest_audio_file(audio_dir)
    if not input_wav_audio:
        print(f"Hiba: Nem található .wav audiófájl a(z) '{audio_dir}' könyvtárban.")
        sys.exit(1)
    print(f"Bemeneti audió (WAV): {input_wav_audio}")

    srt_files = collect_srt_files(upload_dir)
    srt_entries = collect_srt_files(upload_dir)
    if srt_entries:
        print("Feliratfájlok hozzáadásra kijelölve:")
        for srt_path, lang in srt_entries:
            lang_info = lang if lang else "ismeretlen"
            print(f"  - {srt_path} (nyelv: {lang_info})")
    else:
        print("Nem található SRT felirat az upload könyvtárban.")

    os.makedirs(output_dir, exist_ok=True)

    target_sample_rate = get_target_sample_rate(input_video)
    print(f"Cél mintavételezési ráta: {target_sample_rate}Hz")

    processed_aac_path = process_audio_to_raw_aac(input_wav_audio, target_sample_rate)
    print(f"Ideiglenes nyers .aac fájl létrehozva: {processed_aac_path}")

    final_video = add_audio_and_subs_to_video(
        video_file=input_video,
        new_audio_aac_file=processed_aac_path,
        srt_entries=srt_entries,
        language=args.language,
        target_sample_rate=target_sample_rate,
        output_dir=output_dir,
    )

    if os.path.exists(processed_aac_path):
        os.remove(processed_aac_path)
        print(f"Ideiglenes audiófájl törölve: {processed_aac_path}")

    print(f"\nSikeres végrehajtás! A kimeneti fájl itt található: {final_video}")


if __name__ == "__main__":
    main()
