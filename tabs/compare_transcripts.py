# tabs/compare_transcripts.py
import os
import json
import re
from .utils import normalize_text


def compare_transcripts_whisperx(proj_name, workdir="workdir"):
    """
    Compares JSON transcripts and TXT files and returns data for display.
    """
    try:
        transcripts_split_dir = os.path.join(workdir, proj_name, "transcripts_split")
        split_audio_dir = os.path.join(workdir, proj_name, "split_audio")
        translations_dir = os.path.join(workdir, proj_name, "translations")
        sync_dir = os.path.join(workdir, proj_name, "sync")

        if not os.path.exists(transcripts_split_dir):
            return [], "No transcripts_split directory in the project."

        if not os.path.exists(split_audio_dir):
            return [], "No split_audio directory in the project."

        # List JSON files
        json_files = [f for f in os.listdir(transcripts_split_dir) if f.lower().endswith('.json')]

        if not json_files:
            return [], "No JSON files found in transcripts_split directory."

        # Function to extract timestamp
        def get_timestamp(filename):
            match = re.match(r'(\d{2})-(\d{2})-(\d{2}\.\d+)-', filename)
            if match:
                hours, minutes, seconds = match.groups()
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            else:
                return 0  # If no match, return 0

        # Sort json_files based on timestamp
        json_files = sorted(json_files, key=get_timestamp)

        data_list = []

        for json_file in json_files:
            json_path = os.path.join(transcripts_split_dir, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            language = data.get("language", "N/A")
            segments = data.get("segments", [])

            basename = os.path.splitext(json_file)[0]

            # Concatenate all JSON segment texts
            json_full_text = " ".join([segment.get("text", "") for segment in segments]).strip()

            # Normalize JSON text
            json_full_text_normalized = normalize_text(json_full_text)

            # Determine corresponding TXT file
            txt_filename = f"{basename}.txt"
            txt_path = os.path.join(split_audio_dir, txt_filename)

            if not os.path.exists(txt_path):
                txt_text = "TXT file missing."
                txt_text_normalized = "txt file missing."
            else:
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    txt_text = txt_file.read().strip()
                # Normalize TXT text
                txt_text_normalized = normalize_text(txt_text)

            # Compare texts
            match = (json_full_text_normalized == txt_text_normalized)

            # Determine corresponding audio file
            wav_filename = f"{basename}.wav"
            mp3_filename = f"{basename}.mp3"
            wav_path = os.path.join(split_audio_dir, wav_filename)
            mp3_path = os.path.join(split_audio_dir, mp3_filename)

            if os.path.exists(wav_path):
                audio_file_path = wav_path
            elif os.path.exists(mp3_path):
                audio_file_path = mp3_path
            else:
                audio_file_path = None

            # Translated TXT content
            translated_txt_filename = f"{basename}.txt"
            translated_txt_path = os.path.join(translations_dir, translated_txt_filename)

            if not os.path.exists(translated_txt_path):
                translated_txt_content = "Translation not yet done."
            else:
                with open(translated_txt_path, 'r', encoding='utf-8') as translated_txt_file:
                    translated_txt_content = translated_txt_file.read().strip()

            # Sync audio file
            sync_wav_filename = f"{basename}.wav"
            sync_wav_path = os.path.join(sync_dir, sync_wav_filename)

            if os.path.exists(sync_wav_path):
                sync_audio_file_path = sync_wav_path
            else:
                sync_audio_file_path = None

            # Collect data
            data_item = {
                'language': language,
                'json_text': json_full_text,
                'txt_text': txt_text,
                'audio_file': audio_file_path,
                'translated_txt': translated_txt_content,
                'translated_txt_path': translated_txt_path,  # New field
                'sync_audio_file': sync_audio_file_path,
                'match': match
            }

            data_list.append(data_item)

        return data_list, ""

    except Exception as e:
        return [], f"An error occurred during comparison: {str(e)}"
