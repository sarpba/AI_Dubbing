# tabs/project_creation.py
import os
import shutil
import subprocess
from moviepy import VideoFileClip #.editor
from .utils import run_script, ensure_directory

def upload_and_extract_audio(proj_name, video_path, workdir="workdir"):
    """
    Feltölti a videófájlt, létrehozza a projekt struktúráját, és kivonja az audiót valamint a feliratokat.

    Args:
        proj_name (str): A projekt neve.
        video_path (str): A feltöltött videófájl elérési útja.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Returns:
        str: Az eredmény üzenete.
    """
    try:
        # Ellenőrizzük, hogy egy fájl lett-e feltöltve
        if video_path is None:
            return "Nincs feltöltött fájl."

        # Ellenőrizzük a fájl méretét (max 2000MB)
        file_size = os.path.getsize(video_path)
        max_size = 2000 * 1024 * 1024  # 2000MB
        if file_size > max_size:
            return "A feltöltött fájl mérete meghaladja a 2000MB-ot."

        # Projekt mappa létrehozása
        project_path = os.path.join(workdir, proj_name)
        uploads_path = os.path.join(project_path, "uploads")
        audio_path = os.path.join(project_path, "audio")
        subtitles_path = os.path.join(project_path, "subtitles")
        ensure_directory(uploads_path)
        ensure_directory(audio_path)
        ensure_directory(subtitles_path)

        # Videó másolása az uploads mappába
        video_filename = os.path.basename(video_path)
        dest_video_path = os.path.join(uploads_path, video_filename)
        shutil.copy(video_path, dest_video_path)

        # Audio kivonása
        video = VideoFileClip(dest_video_path)
        audio = video.audio
        audio_file_path = os.path.join(audio_path, f"{os.path.splitext(video_filename)[0]}.mp3")
        audio.write_audiofile(audio_file_path)
        audio.close()
        video.close()

        # Feliratok kinyerése
        # Ellenőrizzük, hogy van-e felirat stream a videóban
        ffprobe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=index,codec_type:stream_tags=language",
            "-of", "csv=p=0",
            dest_video_path
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        subtitle_streams = []
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 2:
                    stream_index = parts[0]
                    codec_type = parts[1]
                    language = parts[2] if len(parts) > 2 and parts[2] else 'unknown'
                    if codec_type.strip() == 'subtitle':
                        subtitle_streams.append({'index': stream_index, 'language': language})

        if not subtitle_streams:
            subtitles_message = "A videó nem tartalmaz feliratokat."
        else:
            subtitles_extracted = []
            language_count = {}  # Számláló az egyes nyelvi kódokhoz
            for stream in subtitle_streams:
                stream_index = stream['index']
                language = stream['language']
                
                # Tisztítjuk a nyelvi kódot, hogy érvényes fájlnév legyen
                language = language.lower()
                allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
                language = ''.join(c for c in language if c in allowed_chars)
                if not language:
                    language = 'unknown'

                # Számlálás az adott nyelvi kódhoz
                if language in language_count:
                    language_count[language] += 1
                else:
                    language_count[language] = 1

                # Ha több felirat van ugyanabban a nyelvben, számozást adunk
                if language_count[language] > 1:
                    subtitle_filename = f"{os.path.splitext(video_filename)[0]}_subtitle_{language}_{language_count[language]}"
                else:
                    subtitle_filename = f"{os.path.splitext(video_filename)[0]}_subtitle_{language}"
                
                subtitle_file_path = os.path.join(subtitles_path, f"{subtitle_filename}.srt")

                # Felirat kinyerése
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",  # Túlírja a meglévő fájlokat
                    "-i", dest_video_path,
                    "-map", f"0:{stream_index}",
                    subtitle_file_path
                ]
                extraction_result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if extraction_result.returncode == 0:
                    subtitles_extracted.append(subtitle_file_path)
                else:
                    # Hibakezelés, ha a felirat kinyerése sikertelen
                    subtitles_extracted.append(f"Hiba a {stream_index}. felirat kinyerésekor.")

            if subtitles_extracted:
                subtitles_message = "Feliratok sikeresen kinyerve:\n" + "\n".join(subtitles_extracted)
            else:
                subtitles_message = "A feliratok kinyerése sikertelen."

        return (f"Videó és audio sikeresen feltöltve és kivonva.\n"
                f"Audio fájl elérhető itt: {audio_file_path}\n"
                f"{subtitles_message}")

    except Exception as e:
        return f"Hiba történt a feltöltés, az audio vagy a feliratok kivonása során: {str(e)}"
