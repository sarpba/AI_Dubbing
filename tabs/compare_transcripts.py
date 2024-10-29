# tabs/compare_transcripts.py
import os
import json
import base64
from .utils import normalize_text, escape_html_text

def compare_transcripts_whisperx(proj_name, workdir="workdir"):
    """
    Összehasonlítja a transcripts_split mappában lévő JSON fájlok és a split_audio mappában lévő TXT fájlok tartalmát.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Returns:
        str: HTML táblázat a összehasonlítás eredményével.
    """
    try:
        transcripts_split_dir = os.path.join(workdir, proj_name, "transcripts_split")
        split_audio_dir = os.path.join(workdir, proj_name, "split_audio")
        translations_dir = os.path.join(workdir, proj_name, "translations")
        sync_dir = os.path.join(workdir, proj_name, "sync")

        if not os.path.exists(transcripts_split_dir):
            return "Nincs transcripts_split könyvtár a projektben."

        if not os.path.exists(split_audio_dir):
            return "Nincs split_audio könyvtár a projektben."

        # List JSON files
        json_files = [f for f in os.listdir(transcripts_split_dir) if f.lower().endswith('.json')]

        if not json_files:
            return "Nincs található JSON fájl a transcripts_split könyvtárban."

        # Initialize HTML table
        html_content = """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                vertical-align: top;
            }
            th {
                background-color: #f2f2f2;
                text-align: left;
            }
            .mismatch {
                background-color: #f8d7da; /* Piros */
            }
            .match {
                background-color: #d4edda; /* Zöld */
            }
        </style>
        <table>
            <tr>
                <th>Nyelvi kód</th>
                <th>Json text tartalma</th>
                <th>TXT tartalma</th>
                <th>Lejátszó</th>
                <th>Lefordított TXT tartalma</th>
                <th>Szinkron darabok</th>
            </tr>
        """

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
                txt_text = "TXT fájl hiányzik."
                txt_text_normalized = "txt fájl hiányzik."
            else:
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    txt_text = txt_file.read().strip()
                # Normalize TXT text
                txt_text_normalized = normalize_text(txt_text)

            # Compare texts
            if json_full_text_normalized == txt_text_normalized:
                row_class = "match"
            else:
                row_class = "mismatch"

            # Determine corresponding audio file
            # Assuming .wav first, else .mp3
            wav_filename = f"{basename}.wav"
            mp3_filename = f"{basename}.mp3"
            wav_path = os.path.join(split_audio_dir, wav_filename)
            mp3_path = os.path.join(split_audio_dir, mp3_filename)

            if os.path.exists(wav_path):
                audio_file_path = wav_path
                audio_ext = "wav"
            elif os.path.exists(mp3_path):
                audio_file_path = mp3_path
                audio_ext = "mp3"
            else:
                audio_file_path = None

            if audio_file_path:
                # Read audio file and encode in base64
                with open(audio_file_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                    encoded_audio = base64.b64encode(audio_data).decode('utf-8')

                if audio_ext == "wav":
                    mime_type = "audio/wav"
                elif audio_ext == "mp3":
                    mime_type = "audio/mpeg"
                else:
                    mime_type = "audio/mpeg"  # default

                audio_src = f"data:{mime_type};base64,{encoded_audio}"
                audio_player = f'<audio controls><source src="{audio_src}" type="{mime_type}">Your browser does not support the audio element.</audio>'
            else:
                audio_player = "N/A"

            # Lefordított TXT tartalma
            translated_txt_filename = f"{basename}.txt"
            translated_txt_path = os.path.join(translations_dir, translated_txt_filename)

            if not os.path.exists(translated_txt_path):
                translated_txt_content = "Még nem készült fordítás"
            else:
                with open(translated_txt_path, 'r', encoding='utf-8') as translated_txt_file:
                    translated_txt_content = translated_txt_file.read().strip()

            # Szinkron darabok
            sync_wav_filename = f"{basename}.wav"
            sync_wav_path = os.path.join(sync_dir, sync_wav_filename)

            if os.path.exists(sync_wav_path):
                # Read audio file and encode in base64
                with open(sync_wav_path, 'rb') as sync_audio_file:
                    sync_audio_data = sync_audio_file.read()
                    sync_encoded_audio = base64.b64encode(sync_audio_data).decode('utf-8')

                sync_audio_src = f"data:audio/wav;base64,{sync_encoded_audio}"
                sync_audio_player = f'<audio controls><source src="{sync_audio_src}" type="audio/wav">Your browser does not support the audio element.</audio>'
            else:
                sync_audio_player = "Még nem készült szinkron"

            # Escape HTML in text to prevent issues
            json_text_escaped = escape_html_text(json_full_text)
            txt_text_escaped = escape_html_text(txt_text)
            translated_txt_escaped = escape_html_text(translated_txt_content)

            # Append row to table
            html_content += f"""
            <tr class="{row_class}">
                <td>{language}</td>
                <td>{json_text_escaped}</td>
                <td>{txt_text_escaped}</td>
                <td>{audio_player}</td>
                <td>{translated_txt_escaped}</td>
                <td>{sync_audio_player}</td>
            </tr>
            """

        html_content += "</table>"

        return html_content

    except Exception as e:
        return f"Hiba történt az összehasonlítás során: {str(e)}"

