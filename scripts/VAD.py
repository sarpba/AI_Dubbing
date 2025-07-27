import os
import wave
import contextlib
import webrtcvad
import json
import argparse
import subprocess
import tempfile

ALLOWED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
TARGET_SAMPLE_RATE = 16000  # célérték: például 16 kHz

def convert_to_allowed_format(path):
    """FFmpeg segítségével konvertálja a fájlt a megfelelő formátumba (mono, TARGET_SAMPLE_RATE)."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), temp_file.name],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        os.remove(temp_file.name)
        raise RuntimeError(f"Konverzió sikertelen: {e}")
    return temp_file.name

def read_wave(path):
    """
    Beolvassa a WAV fájlt.
    Ha a fájl nem mono vagy a mintavételi frekvenciája nem megfelelő,
    akkor ideiglenes fájlt készít a megfelelő formátumba.
    """
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
    except wave.Error as e:
        raise ValueError(f"Hiba a WAV fájl olvasása közben: {e}")

    if channels != 1 or sample_rate not in ALLOWED_SAMPLE_RATES:
        print(f"A(z) {path} fájl nem megfelelő paraméterekkel rendelkezik (csatornák: {channels}, sample rate: {sample_rate}). Konvertálás...")
        converted_path = convert_to_allowed_format(path)
        with contextlib.closing(wave.open(converted_path, 'rb')) as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        os.remove(converted_path)
        return frames, sample_rate, sample_width

    with contextlib.closing(wave.open(path, 'rb')) as wf:
        frames = wf.readframes(wf.getnframes())
    return frames, sample_rate, sample_width

def frame_generator(frame_duration_ms, audio, sample_rate, sample_width):
    """Az audio adatokat rögzített hosszúságú (ms-ben) frame-ekre bontja."""
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)
    offset = 0
    timestamp = 0.0
    frame_duration = frame_duration_ms / 1000.0
    while offset + bytes_per_frame <= len(audio):
        yield audio[offset:offset+bytes_per_frame], timestamp
        timestamp += frame_duration
        offset += bytes_per_frame

def detect_first_speech(audio_path):
    """Az első beszédet tartalmazó frame pontos időpontját határozza meg webrtcvad segítségével."""
    audio, sample_rate, sample_width = read_wave(audio_path)
    vad = webrtcvad.Vad(3)  # 0-3 között, 3 a legszigorúbb
    for frame, timestamp in frame_generator(30, audio, sample_rate, sample_width):
        if vad.is_speech(frame, sample_rate):
            return timestamp
    return None

def process_file(file_path):
    """Feldolgoz egy WAV fájlt, és létrehoz egy JSON fájlt, melyben csak az első szó start ideje szerepel."""
    first_word_start = detect_first_speech(file_path)
    if first_word_start is None:
        print(f"Nem talált beszédet a {file_path} fájlban.")
        return

    output_data = {
        "segments": [
            {
                "start": first_word_start
            }
        ]
    }

    base, _ = os.path.splitext(file_path)
    output_file = base + ".json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Kész: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="VAD alapú JSON generátor WAV fájlokhoz (csak az első szó start idejét menti)."
    )
    # Pozícionális argumentum: könyvtár elérési útja
    parser.add_argument("directory", help="A feldolgozandó WAV fájlokat tartalmazó könyvtár elérési útja.")
    # Ismeretlen argumentumok figyelmen kívül hagyása
    args, unknown = parser.parse_known_args()

    input_dir = args.directory
    if not os.path.isdir(input_dir):
        print(f"A megadott útvonal nem könyvtár: {input_dir}")
        return

    # Feldolgozza a könyvtárban található .wav fájlokat, de átugorja azokat, ahol már van JSON fájl
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            base, _ = os.path.splitext(file_path)
            json_path = base + ".json"
            if os.path.exists(json_path):
                print(f"Átugorva: {file_path} (JSON már létezik)")
                continue
            try:
                process_file(file_path)
            except Exception as e:
                print(f"Hiba történt a {file_path} feldolgozása közben: {e}")

if __name__ == "__main__":
    main()
