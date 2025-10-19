import os
import json
import re # Reguláris kifejezések importálása a "natural sort"-hoz
import subprocess
import hashlib
import shutil
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from flask import Flask, render_template, send_from_directory, jsonify, request, redirect, url_for, session, Response

# --- ALAP KONFIGURÁCIÓ ÉS ALKALMAZÁS INICIALIZÁLÁSA ---

USERNAME = 'admin'
PASSWORD = 'password123dsvsfvefvaívgbsnreyhgvwevgsen'
TEMP_FOLDER = '/dev/shm/video_viewer_temp'
app = Flask(__name__)
app.secret_key = 'csereld_le_ezt_egy_sokkal_biztonsagosabb_kulcsra_a_valosagban'


# --- INDÍTÁSI FÜGGVÉNYEK ---

def setup_logging():
    access_logger = logging.getLogger('access_logger')
    access_logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('access.log', maxBytes=1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    access_logger.addHandler(handler)

def setup_temp_folder():
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

setup_logging()
setup_temp_folder()


# --- SEGÉDFÜGGVÉNYEK ÉS DEKORÁTOROK ---

def load_users():
    try:
        with open('users.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# === ÚJ SEGÉDFÜGGVÉNY A TERMÉSZETES RENDEZÉSHEZ ===
def natural_sort_key(s):
    """
    Létrehoz egy "kulcsot" a rendezéshez, ami a számokat számként,
    a szöveget szövegként kezeli.
    Pl. "epizod10.mkv" -> ['epizod', 10, '.mkv']
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def find_video_files():
    """
    Rekurzívan megkeresi a videófájlokat. A csoportosítást a 'workdir'
    legfelső szintű mappái alapján végzi, és a fájlokat természetes
    sorrendbe rendezi.
    """
    try:
        with open('config.json', 'r') as f: config = json.load(f)
    except FileNotFoundError: return {}
    workdir_root_name=config.get("DIRECTORIES",{}).get("workdir","workdir")
    download_folder_name=config.get("PROJECT_SUBDIRS",{}).get("download","8_Download")
    video_extensions=('.mp4','.mkv','.avi','.mov')
    grouped_videos={"__main__":[]}
    search_root_path=os.path.abspath(workdir_root_name)
    if not os.path.isdir(search_root_path): return {}
    for current_path,_,files in os.walk(search_root_path):
        if os.path.basename(current_path)==download_folder_name:
            for filename in files:
                if filename.lower().endswith(video_extensions):
                    full_video_path=os.path.join(current_path,filename)
                    relative_to_workdir=os.path.relpath(full_video_path,search_root_path)
                    path_parts=relative_to_workdir.split(os.sep)
                    group_name="__main__"
                    if len(path_parts)>1: group_name=path_parts[0]
                    if group_name not in grouped_videos: grouped_videos[group_name]=[]
                    grouped_videos[group_name].append({"path":os.path.relpath(full_video_path,start=os.getcwd()).replace("\\","/"),"name":filename})
    
    # === MÓDOSÍTÁS: A FÁJLOK RENDEZÉSE MINDEN CSOPORTON BELÜL ===
    for group_name in grouped_videos:
        # A lista elemeit (amik szótárak) a 'name' kulcsuk alapján rendezzük,
        # a natural_sort_key segédfüggvény használatával.
        grouped_videos[group_name].sort(key=lambda item: natural_sort_key(item['name']))

    if not grouped_videos["__main__"]: del grouped_videos["__main__"]
    return grouped_videos


# --- FLASK ÚTVONALAK ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('viewer_page'))
        else:
            error = 'Hibás felhasználónév vagy jelszó.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
@login_required
def viewer_page():
    return render_template('viewer.html')

@app.route('/videos')
@login_required
def get_videos():
    return jsonify(find_video_files())

@app.route('/audio-tracks/<path:filepath>')
@login_required
def get_audio_tracks(filepath):
    if not os.path.exists(filepath): return jsonify({"error": "A fájl nem található"}), 404
    try:
        command=['ffprobe','-v','quiet','-print_format','json','-show_streams','-select_streams','a',filepath]
        result=subprocess.run(command,capture_output=True,text=True,check=True)
        streams=json.loads(result.stdout).get('streams',[])
        tracks=[{'id':s['index'],'label':s.get('tags',{}).get('language',f"Sáv {s['index']}")} for s in streams]
        return jsonify(tracks)
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route('/generate_video', methods=['POST'])
@login_required
def generate_video():
    data = request.get_json()
    source_path = data.get('path')
    track_index = data.get('track_index')
    if not source_path or track_index is None: return jsonify({"error": "Hiányzó paraméterek"}), 400
    
    access_logger = logging.getLogger('access_logger')
    username = session.get('username', 'Ismeretlen')
    ip_addr = request.remote_addr
    log_message = f"User: '{username}' [{ip_addr}] - Fájl: '{source_path}' - Hangsáv index: {track_index}"
    access_logger.info(log_message)

    full_source_path=os.path.abspath(source_path)
    try:
        ffprobe_command=['ffprobe','-v','quiet','-print_format','json','-show_streams',full_source_path]
        result=subprocess.run(ffprobe_command,check=True,capture_output=True,text=True)
        all_streams=json.loads(result.stdout).get('streams',[])
        audio_codec_name=None
        for stream in all_streams:
            if stream['codec_type']=='audio' and stream['index']==track_index:
                audio_codec_name=stream.get('codec_name')
                break
        safe_codecs=['aac','mp3', 'ac3']
        if audio_codec_name in safe_codecs:
            audio_codec_option='copy'
        else:
            audio_codec_option='aac'
    except Exception:
        audio_codec_option='aac'

    unique_id=hashlib.md5(f"{full_source_path}{track_index}".encode()).hexdigest()
    output_filename=f"{unique_id}.mp4"
    output_path=os.path.join(TEMP_FOLDER,output_filename)
    if os.path.exists(output_path):
        return jsonify({"temp_file":output_filename})

    ffmpeg_command=['ffmpeg','-i',full_source_path,'-map','0:v:0','-map',f'0:{track_index}','-c:v','copy','-c:a',audio_codec_option,'-b:a','192k',output_path]
    try:
        subprocess.run(ffmpeg_command,check=True,capture_output=True,text=True)
        return jsonify({"temp_file":output_filename})
    except subprocess.CalledProcessError as e:
        return jsonify({"error":"Hiba a videó feldolgozása közben","details":e.stderr}),500

@app.route('/temp/<filename>')
@login_required
def serve_temp_file(filename):
    return send_from_directory(TEMP_FOLDER, filename)


# --- ALKALMAZÁS INDÍTÁSA ---
if __name__ == '__main__':
    print("Videólejátszó szerver indítása a http://0.0.0.0:5555 címen...")
    app.run(host='0.0.0.0', port=5555, debug=True)
