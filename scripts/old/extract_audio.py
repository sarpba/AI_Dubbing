import os
import sys
import subprocess

def extract_audio(video_path, audio_path):
    try:
        # FFmpeg paranccsal kinyerjük az audiót
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',              # Csak audió
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '44100',     # 44.1kHz mintavétel
            '-ac', '2',        # Sztereó
            '-y',              # Felülírás engedélyezése
            audio_path
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audió kinyerése közben: {e}")
        return False
    except Exception as e:
        print(f"Váratlan hiba: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Használat: python extract_audio.py <input_video> <output_audio>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    if not os.path.exists(video_path):
        print(f"A videófájl nem található: {video_path}")
        sys.exit(1)
        
    success = extract_audio(video_path, audio_path)
    sys.exit(0 if success else 1)
