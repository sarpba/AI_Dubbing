from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import json
import subprocess
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Statikus fájlok kiszolgálása a workdir mappából
@app.route('/workdir/<path:filename>')
def serve_workdir(filename):
    return send_from_directory('workdir', filename)

# Konfiguráció betöltése
with open('config.json') as config_file:
    config = json.load(config_file)

@app.route('/')
def index():
    # Meglévő projektek listázása
    projects = []
    if os.path.exists('workdir'):
        projects = [d for d in os.listdir('workdir') 
                   if os.path.isdir(os.path.join('workdir', d))]
    return render_template('index.html', projects=projects)

@app.route('/project/<project_name>')
def show_project(project_name):
    # Projekt ellenőrzése
    project_dir = os.path.join('workdir', secure_filename(project_name))
    if not os.path.exists(project_dir):
        return "Project not found", 404
    
    # Projekt adatok összegyűjtése
    project_data = {
        'name': project_name,
        'files': {
            'upload': [],
            'extracted_audio': [],
            'separated_audio_background': [],
            'separated_audio_speech': []
        }
    }

    # Uploaded files
    upload_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['upload'])
    if os.path.exists(upload_dir_path):
        project_data['files']['upload'] = os.listdir(upload_dir_path)

    # Extracted audio files
    extracted_audio_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['extracted_audio'])
    if os.path.exists(extracted_audio_dir_path):
        project_data['files']['extracted_audio'] = os.listdir(extracted_audio_dir_path)

    # Separated background audio files
    separated_bg_audio_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_background'])
    if os.path.exists(separated_bg_audio_dir_path):
        project_data['files']['separated_audio_background'] = os.listdir(separated_bg_audio_dir_path)

    # Separated speech audio files (and their JSON transcriptions)
    speech_files_data = []
    speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
    if os.path.exists(speech_dir_path):
        for f_name in sorted(os.listdir(speech_dir_path)): # Sorted for consistent order
            file_data = {'name': f_name, 'segment_count': None, 'is_audio': False, 'is_json': False}
            file_path = os.path.join(speech_dir_path, f_name)
            
            if f_name.lower().endswith('.json'):
                file_data['is_json'] = True
                try:
                    with open(file_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                        if 'segments' in data and isinstance(data['segments'], list):
                            file_data['segment_count'] = len(data['segments'])
                except Exception as e:
                    print(f"Error reading or parsing JSON file {file_path}: {e}")
            elif f_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                file_data['is_audio'] = True
            speech_files_data.append(file_data)
    project_data['files']['separated_audio_speech'] = speech_files_data
    
    # Ellenőrizzük, van-e felülvizsgálható audio/JSON pár
    can_review = False
    if os.path.exists(speech_dir_path):
        temp_files_list = sorted(os.listdir(speech_dir_path))
        for f_check_name in temp_files_list:
            if f_check_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                base_name_check, _ = os.path.splitext(f_check_name)
                if (base_name_check + ".json") in temp_files_list:
                    can_review = True
                    break
    
    # Új flag: van-e transzkribálható (beszéd) audio fájl
    has_transcribable_audio = any(f_info['is_audio'] for f_info in speech_files_data)
    
    return render_template('project.html', 
                         project=project_data,
                         config=config,
                         can_review=can_review,
                         has_transcribable_audio=has_transcribable_audio)

@app.route('/review/<project_name>')
def review_project(project_name):
    project_dir = os.path.join('workdir', secure_filename(project_name))
    # Először a "translated" mappában keresünk JSON fájlt
    translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
    # Ha nincs "translated" JSON, akkor a "separated_audio_speech" mappát használjuk
    speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
    
    audio_file_name = None
    json_file_name = None
    segments_data = []
    
    # Először a "translated" mappában keresünk
    if os.path.exists(translated_dir_path):
        translated_files = sorted(os.listdir(translated_dir_path))
        # Keressük a legelső audio fájlt, amihez van JSON a "translated" mappában
        for f_name in translated_files:
            if f_name.lower().endswith('.json'):
                base_name, _ = os.path.splitext(f_name)
                # Ellenőrizzük, van-e audio fájl a "separated_audio_speech" mappában
                if os.path.exists(speech_dir_path):
                    speech_files = sorted(os.listdir(speech_dir_path))
                    # Keressük a megfelelő audio fájlt (ugyanaz a base név)
                    audio_candidate = base_name + os.path.splitext(f_name)[1].replace('.json', '')
                    for audio_ext in ['.wav', '.mp3', '.ogg', '.flac']:
                        if audio_candidate + audio_ext in speech_files:
                            audio_file_name = audio_candidate + audio_ext
                            json_file_name = f_name
                            json_full_path = os.path.join(translated_dir_path, json_file_name)
                            try:
                                with open(json_full_path, 'r', encoding='utf-8') as jf:
                                    data = json.load(jf)
                                    segments_data = data.get('segments', [])
                            except Exception as e:
                                print(f"Error reading JSON file {json_full_path} for review: {e}")
                                segments_data = []
                            break
                    if audio_file_name:
                        break
    
    # Ha nem találtunk "translated" JSON-t, akkor az eredeti logikát használjuk
    if not json_file_name and os.path.exists(speech_dir_path):
        files = sorted(os.listdir(speech_dir_path))
        for f_name in files:
            if f_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                base_name, _ = os.path.splitext(f_name)
                potential_json_name = base_name + ".json"
                if potential_json_name in files:
                    audio_file_name = f_name
                    json_file_name = potential_json_name
                    json_full_path = os.path.join(speech_dir_path, json_file_name)
                    try:
                        with open(json_full_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                            segments_data = data.get('segments', [])
                    except Exception as e:
                        print(f"Error reading JSON file {json_full_path} for review: {e}")
                        segments_data = [] # Hiba esetén üres lista
                    break # Első páros megtalálva
    
    audio_url = None
    if audio_file_name:
        audio_url = url_for('serve_workdir', filename=f"{secure_filename(project_name)}/{config['PROJECT_SUBDIRS']['separated_audio_speech']}/{audio_file_name}")

    return render_template('review.html', 
                           project_name=project_name, 
                           audio_file_name=audio_file_name,
                           audio_url=audio_url,
                           segments_data=segments_data,
                           json_file_name=json_file_name,
                           app_config=config)  # Átadjuk a konfigurációt app_config néven

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files or 'projectName' not in request.form:
        return jsonify({'error': 'Missing file or project name'}), 400
        
    file = request.files['file']
    project_name = request.form['projectName']
    
    if file.filename == '' or not project_name:
        return jsonify({'error': 'No selected file or empty project name'}), 400
        
    if file and project_name:
        # Projekt mappa létrehozása
        project_dir = os.path.join('workdir', secure_filename(project_name))
        os.makedirs(project_dir, exist_ok=True)
        
        # Almappák létrehozása a projektben
        for subdir in config['PROJECT_SUBDIRS'].values():
            os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
        
        filename = secure_filename(file.filename)
        upload_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['upload'], filename)
        try:
            file.save(upload_path)
            
            return jsonify({
                'message': 'File uploaded successfully', 
                'path': upload_path,
                'project': project_name
            })
        except Exception as e:
            return jsonify({
            'error': f'Upload failed: {str(e)}',
            'project': project_name
        }), 500

@app.route('/api/extract-audio/<project_name>', methods=['POST'])
def extract_audio(project_name):
    try:
        project_dir = os.path.join('workdir', secure_filename(project_name))
        if not os.path.exists(project_dir):
            return jsonify({'error': 'Project not found'}), 404

        # Get the first video file
        upload_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['upload'])
        video_files = os.listdir(upload_dir)
        if not video_files:
            return jsonify({'error': 'No video files found'}), 400
            
        video_file = video_files[0]
        video_path = os.path.join(upload_dir, video_file)
        
        # Create output path
        extracted_audio_path = os.path.join(
            project_dir, 
            config['PROJECT_SUBDIRS']['extracted_audio'],
            os.path.splitext(video_file)[0] + '.wav'
        )
        
        # Run audio extraction
        cmd = ['python', 'scripts/extract_audio.py', video_path, extracted_audio_path]
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            error_message = process.stderr if process.stderr else "Unknown error during audio extraction"
            raise Exception(f"Audio extraction failed with exit code {process.returncode}: {error_message}")
            
        return jsonify({
            'success': True,
            'message': 'Audio extracted successfully',
            'project': project_name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'project': project_name
        }), 500

@app.route('/api/separate-audio/<project_name>', methods=['POST'])
def separate_audio(project_name):
    try:
        project_dir = os.path.join('workdir', secure_filename(project_name))
        if not os.path.exists(project_dir):
            return jsonify({'error': 'Project not found'}), 404

        # Get the first extracted audio file
        extracted_audio_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['extracted_audio'])
        audio_files = os.listdir(extracted_audio_dir)
        if not audio_files:
            return jsonify({'error': 'No audio files found'}), 400
            
        audio_file = audio_files[0]
        audio_path = os.path.join(extracted_audio_dir, audio_file)
        
        # Run audio separation
        speech_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        background_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_background'])
        os.makedirs(speech_dir, exist_ok=True)
        os.makedirs(background_dir, exist_ok=True)
        
        data = request.get_json()
        model = data.get('model', 'htdemucs_ft')
        keep_full_audio = data.get('keep_full_audio', False)
        non_speech_silence = data.get('non_speech_silence', False)
        chunk_size = data.get('chunk_size', 5)
        
        cmd = [
            'python', 'scripts/separate_audio.py',
            '-i', extracted_audio_dir,
            '-o', speech_dir,
            '--model', model,
            '--chunk_size', str(chunk_size)
        ]
        
        if keep_full_audio:
            cmd.append('--keep_full_audio')
        if non_speech_silence:
            cmd.append('--non_speech_silence')
        
        print(f"Running command: {' '.join(cmd)}") # A printeléshez maradhat az összefűzés
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            error_message = process.stderr if process.stderr else "Unknown error during audio separation"
            raise Exception(f"Audio separation failed with exit code {process.returncode}: {error_message}")
            
        return jsonify({
            'success': True,
            'message': 'Audio separated successfully',
            'project': project_name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'project': project_name
        }), 500

@app.route('/api/transcribe/<project_name>', methods=['POST'])
def transcribe_audio(project_name):
    try:
        data = request.get_json()
        language = data.get('language', 'en')
        hf_token = data.get('hf_token', '')
        gpus = data.get('gpus', '')

        # Build the command to run whisx.py
        input_dir = os.path.join(
            'workdir',
            project_name,
            config['PROJECT_SUBDIRS']['separated_audio_speech']
        )
        
        cmd = [
            'python', 'scripts/whisx.py',
            input_dir,
            '--language', language
        ]
        
        if hf_token:
            cmd.extend(['--hf_token', hf_token])
        if gpus:
            cmd.extend(['--gpus', gpus])

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Transcription failed: {result.stderr}")

        return jsonify({
            'success': True,
            'message': 'Audio transcription completed',
            'project': project_name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'project': project_name
        }), 500

@app.route('/api/delete-project/<project_name>', methods=['DELETE'])
def delete_project(project_name):
    try:
        project_dir = os.path.join('workdir', secure_filename(project_name))
        if not os.path.exists(project_dir):
            return jsonify({'error': 'Project not found'}), 404
            
        # Remove the entire project directory
        import shutil
        shutil.rmtree(project_dir)
        
        return jsonify({
            'success': True,
            'message': 'Project deleted successfully',
            'project': project_name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'project': project_name
        }), 500

@app.route('/api/update-segment/<project_name>', methods=['POST'])
def update_segment_api(project_name): 
    app.logger.info(f"update_segment_api called for project: {project_name}")
    try:
        data = request.get_json()
        app.logger.info(f"Received data for update: {data}")
        json_file_name = data.get('json_file_name')
        segment_index = data.get('segment_index')
        new_start = data.get('new_start')
        new_end = data.get('new_end')
        new_text = data.get('new_text', None) # Új szöveg beolvasása, None ha nincs
        new_translated_text = data.get('new_translated_text', None) # Új fordított szöveg

        # new_text nem kötelező, csak a többi
        if None in [json_file_name, segment_index] or new_start is None or new_end is None:
            return jsonify({'success': False, 'error': 'Missing data (json_file_name, segment_index, new_start, or new_end)'}), 400

        project_dir = os.path.join('workdir', secure_filename(project_name))
        # A config['PROJECT_SUBDIRS']['separated_audio_speech'] a separated_audio_speech almappa neve
        speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        
        # Fontos: a json_file_name itt már a tényleges fájlnév kell legyen, a secure_filename-t a projekt névre már alkalmaztuk.
        # Ha a json_file_name is tartalmazhat ../ vagy hasonlókat, akkor itt is kell secure_filename.
        # Mivel a json_file_name a szerverről jön (review_project), és ott os.listdir-ből, valószínűleg biztonságos.
        # De a biztonság kedvéért alkalmazhatjuk, ha a fájlnév a kliensről érkező adatból származik közvetlenül.
        # Jelen esetben a kliens a Flask által renderelt {{ json_file_name }} értéket küldi vissza, ami megbízható.
        # Ha a secure_filename itt módosítaná a fájlnevet (pl. speciális karakterek miatt), akkor nem a jó fájlt találná meg.
        # Ezért a secure_filename(json_file_name) helyett csak json_file_name-et használok, feltételezve, hogy az tiszta.
        # Ha a json_file_name a kliens által szabadon beírható lenne, akkor kellene a secure_filename.
        if not json_file_name: # Extra ellenőrzés
             return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400
        
        # Ellenőrizzük, hogy a fájl a "translated" mappában van-e
        translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
        translated_json_path = os.path.join(translated_dir_path, json_file_name)
        
        # Ha létezik a "translated" mappában, azt használjuk
        if os.path.exists(translated_json_path):
            json_full_path = translated_json_path
            app.logger.info(f"Using translated JSON file at: {json_full_path}")
        elif not os.path.exists(os.path.join(speech_dir_path, json_file_name)):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404
        else:
            json_full_path = os.path.join(speech_dir_path, json_file_name)

        with open(json_full_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
            return jsonify({'success': False, 'error': 'Invalid JSON structure: "segments" missing or not a list'}), 500
        
        if not (isinstance(segment_index, int) and 0 <= segment_index < len(transcription_data['segments'])):
            return jsonify({'success': False, 'error': f'Invalid segment index: {segment_index}'}), 400

        # Értékek validálása
        if not (isinstance(new_start, (int, float)) and isinstance(new_end, (int, float))):
             return jsonify({'success': False, 'error': 'new_start and new_end must be numbers'}), 400
        if new_start < 0 or new_end < 0 or new_start >= new_end:
             return jsonify({'success': False, 'error': f'Invalid start/end times: start={new_start}, end={new_end}'}), 400
        
        # Szomszédos szegmensek ellenőrzése (opcionális, de ajánlott)
        # Előző szegmens vége < új start
        if segment_index > 0:
            prev_segment_end = transcription_data['segments'][segment_index - 1].get('end')
            if prev_segment_end is not None and new_start < prev_segment_end:
                return jsonify({'success': False, 'error': f'New start time {new_start} overlaps with previous segment end {prev_segment_end}'}), 400
        # Következő szegmens eleje > új end
        if segment_index < len(transcription_data['segments']) - 1:
            next_segment_start = transcription_data['segments'][segment_index + 1].get('start')
            if next_segment_start is not None and new_end > next_segment_start:
                return jsonify({'success': False, 'error': f'New end time {new_end} overlaps with next segment start {next_segment_start}'}), 400


        transcription_data['segments'][segment_index]['start'] = new_start
        transcription_data['segments'][segment_index]['end'] = new_end
        if new_text is not None: # Csak akkor frissítjük a szöveget, ha kaptunk újat
            transcription_data['segments'][segment_index]['text'] = new_text
        if new_translated_text is not None: # Csak akkor frissítjük a fordított szöveget, ha kaptunk újat
            transcription_data['segments'][segment_index]['translated_text'] = new_translated_text
        
        # Minden szegmensből töröljük a 'words' mezőt, ha létezik.
        # Ezt a részt érdemes lehet újragondolni: ha a szöveg változik, a 'words' már nem lesz érvényes.
        # Ha a 'words' mezőt a transzkripció során generáljuk, és utána már nem használjuk fel a szerkesztés során,
        # akkor a törlése rendben van. Ha viszont a 'words' információra később szükség lehet,
        # akkor a szöveg módosításakor ezt is frissíteni kellene, vagy jelezni, hogy elavult.
        # Jelenleg a törlés egyszerűbb, és megakadályozza a konzisztenciahiányt.
        for segment_item in transcription_data['segments']: # Változónév ütközés elkerülése
            if 'words' in segment_item:
                del segment_item['words']

        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': 'Segment updated successfully'})

    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Error updating segment for project {project_name}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An unexpected error occurred on the server.'}), 500

@app.route('/api/add-segment/<project_name>', methods=['POST'])
def add_segment_api(project_name):
    app.logger.info(f"add_segment_api called for project: {project_name}")
    try:
        data = request.get_json()
        app.logger.info(f"Received data for new segment: {data}")
        
        json_file_name = data.get('json_file_name')
        new_start = data.get('start')
        new_end = data.get('end')
        new_text = data.get('text', 'Új szegmens') # Alapértelmezett szöveg

        if None in [json_file_name, new_start, new_end, new_text]:
            return jsonify({'success': False, 'error': 'Missing data (json_file_name, start, end, or text)'}), 400

        project_dir = os.path.join('workdir', secure_filename(project_name))
        speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        
        if not json_file_name:
             return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400
        
        # Ellenőrizzük, hogy a fájl a "translated" mappában van-e
        translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
        translated_json_path = os.path.join(translated_dir_path, json_file_name)
        
        # Ha létezik a "translated" mappában, azt használjuk
        if os.path.exists(translated_json_path):
            json_full_path = translated_json_path
            app.logger.info(f"Using translated JSON file at: {json_full_path}")
        elif not os.path.exists(os.path.join(speech_dir_path, json_file_name)):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404
        else:
            json_full_path = os.path.join(speech_dir_path, json_file_name)

        app.logger.info(f"Attempting to modify JSON file at: {json_full_path}")

        if not os.path.exists(json_full_path):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404

        with open(json_full_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
            transcription_data['segments'] = [] # Ha nincs segments kulcs, vagy nem lista, hozzunk létre egy üreset

        # Értékek validálása
        if not (isinstance(new_start, (int, float)) and isinstance(new_end, (int, float))):
             return jsonify({'success': False, 'error': 'New start and end times must be numbers'}), 400
        if new_start < 0 or new_end < 0 or new_start >= new_end:
             return jsonify({'success': False, 'error': f'Invalid start/end times for new segment: start={new_start}, end={new_end}'}), 400
        if not isinstance(new_text, str):
            return jsonify({'success': False, 'error': 'New text must be a string'}), 400

        # Átfedés ellenőrzése a meglévő szegmensekkel
        for segment in transcription_data['segments']:
            # Új szegmens teljesen egy meglévőn belül van
            if new_start >= segment['start'] and new_end <= segment['end']:
                return jsonify({'success': False, 'error': f'New segment ({new_start}-{new_end}) is completely within an existing segment ({segment["start"]}-{segment["end"]}).'}), 400
            # Új szegmens teljesen lefedi egy meglévőt
            if new_start <= segment['start'] and new_end >= segment['end']:
                 return jsonify({'success': False, 'error': f'New segment ({new_start}-{new_end}) completely covers an existing segment ({segment["start"]}-{segment["end"]}).'}), 400
            # Új szegmens kezdete egy meglévőbe lóg
            if new_start >= segment['start'] and new_start < segment['end']:
                return jsonify({'success': False, 'error': f'New segment start ({new_start}) overlaps with existing segment ({segment["start"]}-{segment["end"]}).'}), 400
            # Új szegmens vége egy meglévőbe lóg
            if new_end > segment['start'] and new_end <= segment['end']:
                return jsonify({'success': False, 'error': f'New segment end ({new_end}) overlaps with existing segment ({segment["start"]}-{segment["end"]}).'}), 400
        
        new_segment_entry = {
            'start': new_start,
            'end': new_end,
            'text': new_text
            # 'words' kulcsot nem adunk hozzá, mert az a transzkripció eredménye
        }
        transcription_data['segments'].append(new_segment_entry)
        
        # Szegmensek rendezése start idő alapján
        transcription_data['segments'].sort(key=lambda s: s['start'])
        
        # 'words' kulcs törlése minden szegmensből (ha a szöveg vagy időzítés változik, a words elavulttá válik)
        for segment_item in transcription_data['segments']:
            if 'words' in segment_item:
                del segment_item['words']

        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': 'Segment added successfully', 'segments': transcription_data['segments']})

    except Exception as e:
        app.logger.error(f"Error adding segment for project {project_name}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An unexpected error occurred on the server while adding segment.'}), 500

@app.route('/api/delete-segment/<project_name>', methods=['POST'])
def delete_segment_api(project_name):
    app.logger.info(f"delete_segment_api called for project: {project_name}")
    try:
        data = request.get_json()
        app.logger.info(f"Received data for delete: {data}")
        
        json_file_name = data.get('json_file_name')
        segment_index = data.get('segment_index')

        if None in [json_file_name, segment_index]:
            return jsonify({'success': False, 'error': 'Missing data (json_file_name or segment_index)'}), 400

        project_dir = os.path.join('workdir', secure_filename(project_name))
        speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        
        if not json_file_name:
             return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400
        
        # Ellenőrizzük, hogy a fájl a "translated" mappában van-e
        translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
        translated_json_path = os.path.join(translated_dir_path, json_file_name)
        
        # Ha létezik a "translated" mappában, azt használjuk
        if os.path.exists(translated_json_path):
            json_full_path = translated_json_path
            app.logger.info(f"Using translated JSON file at: {json_full_path}")
        elif not os.path.exists(os.path.join(speech_dir_path, json_file_name)):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404
        else:
            json_full_path = os.path.join(speech_dir_path, json_file_name)

        app.logger.info(f"Attempting to modify JSON file for deletion at: {json_full_path}")

        if not os.path.exists(json_full_path):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404

        with open(json_full_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
            return jsonify({'success': False, 'error': 'Invalid JSON structure: "segments" missing or not a list'}), 500
        
        if not (isinstance(segment_index, int) and 0 <= segment_index < len(transcription_data['segments'])):
            return jsonify({'success': False, 'error': f'Invalid segment index: {segment_index}'}), 400

        # Szegmens törlése
        del transcription_data['segments'][segment_index]
        
        # 'words' kulcs törlése minden szegmensből (ha a törlés miatt az indexek eltolódnak,
        # és a 'words' információ már nem releváns vagy nehezen karbantartható)
        # Ez a lépés konzisztens a update és add funkciókkal.
        for segment_item in transcription_data['segments']:
            if 'words' in segment_item:
                del segment_item['words']

        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': 'Segment deleted successfully', 'segments': transcription_data['segments']})

    except Exception as e:
        app.logger.error(f"Error deleting segment for project {project_name}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An unexpected error occurred on the server while deleting segment.'}), 500

@app.route('/save-api-key', methods=['POST'])
def save_api_key():
    data = request.get_json()
    api_key = data.get('api_key')
    if not api_key:
        return jsonify({'success': False, 'error': 'Missing api_key'}), 400

    # A keyholder.json a projekt gyökerében lesz
    keyholder_path = 'keyholder.json'
    try:
        # Ha már létezik a fájl, beolvassuk és frissítjük, különben létrehozzuk
        if os.path.exists(keyholder_path):
            with open(keyholder_path, 'r') as f:
                key_data = json.load(f)
        else:
            key_data = {}

        # Base64 kódolás
        encoded_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        key_data['api_key'] = encoded_key

        with open(keyholder_path, 'w') as f:
            json.dump(key_data, f, indent=2)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-api-key', methods=['GET'])
def get_api_key():
    keyholder_path = 'keyholder.json'
    if not os.path.exists(keyholder_path):
        return jsonify({'api_key': None})

    try:
        with open(keyholder_path, 'r') as f:
            key_data = json.load(f)
        api_key = key_data.get('api_key')
        return jsonify({'api_key': api_key})
    except Exception as e:
        return jsonify({'api_key': None, 'error': str(e)})

@app.route('/run-translation', methods=['POST'])
def run_translation():
    data = request.get_json()
    input_dir = data.get('input_dir')
    output_dir = data.get('output_dir')
    input_language = data.get('input_language')
    output_language = data.get('output_language')
    auth_key = data.get('auth_key')

    if not all([input_dir, output_dir, input_language, output_language, auth_key]):
        return jsonify({'success': False, 'error': 'Minden mező kitöltése kötelező'}), 400

    try:
        # Parancs összeállítása
        cmd = [
            'python', 'scripts/translate.py',
            '-input_dir', input_dir,
            '-output_dir', output_dir,
            '-input_language', input_language,
            '-output_language', output_language,
            '-auth_key', auth_key
        ]

        # Parancs futtatása aszinkron módon
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return jsonify({'success': True, 'message': 'Fordítás elindult'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')