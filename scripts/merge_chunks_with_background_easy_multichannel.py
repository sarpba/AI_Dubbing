import os
import re
import argparse
import json
from datetime import datetime
from pydub import AudioSegment
import multiprocessing

# A szkript eleje (a segédfüggvények) változatlan
def parse_time_string(time_str):
    pattern = r"^(\d{2})-(\d{2})-(\d{2})-(\d{3})$"
    match = re.match(pattern, time_str)
    if not match:
        raise ValueError(f"Time string '{time_str}' does not match the expected format HH-MM-SS-mmm.")
    h, m, s, ms = match.groups()
    total_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)
    return total_ms

def parse_filename(filename):
    name_without_ext, _ = os.path.splitext(filename)
    parts = name_without_ext.split('_')
    if len(parts) < 2:
        raise ValueError(f"Filename '{filename}' does not contain enough parts separated by underscores.")
    start_time_str = parts[0]
    end_time_str = parts[1]
    try:
        start_time = parse_time_string(start_time_str)
        end_time = parse_time_string(end_time_str)
    except ValueError as e:
        raise ValueError(f"Error parsing filename '{filename}': {e}")
    return start_time, end_time

def load_chunk(filepath):
    try:
        filename = os.path.basename(filepath)
        start_time, end_time = parse_filename(filename)
        audio = AudioSegment.from_wav(filepath)
        expected_duration = end_time - start_time
        actual_duration = len(audio)
        if abs(expected_duration - actual_duration) > 10:
            print(f"Warning: Duration mismatch in '{filename}'. Expected {expected_duration} ms, got {actual_duration} ms.")
        return (start_time, end_time, audio)
    except (ValueError, FileNotFoundError) as ve:
        print(f"Skipping file: {ve}")
        return None

def merge_chunk_group(group_data):
    chunk_list, max_end_time = group_data
    group_timeline = AudioSegment.silent(duration=max_end_time)
    for start_time, _, audio in chunk_list:
        group_timeline = group_timeline.overlay(audio, position=start_time)
    return group_timeline

def create_dubbed_timeline_fully_parallel(input_folder):
    wav_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
    if not wav_files:
        print("No valid WAV files found to merge.")
        return None
    num_processes = multiprocessing.cpu_count()
    print(f"Phase 1: Loading {len(wav_files)} audio chunks in parallel using {num_processes} processes...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(load_chunk, wav_files)
    print("Phase 1 complete: All chunks loaded into memory.")
    timeline_data = sorted([r for r in results if r is not None], key=lambda x: x[0])
    if not timeline_data:
        print("No valid WAV files could be processed.")
        return None
    max_end_time = max(item[1] for item in timeline_data)
    print(f"\nPhase 2: Merging {len(timeline_data)} chunks into a final timeline in parallel...")
    chunk_size = (len(timeline_data) + num_processes - 1) // num_processes
    chunk_groups = [timeline_data[i:i + chunk_size] for i in range(0, len(timeline_data), chunk_size)]
    grouped_data_for_pool = [(group, max_end_time) for group in chunk_groups if group]
    with multiprocessing.Pool(processes=num_processes) as pool:
        partial_timelines = pool.map(merge_chunk_group, grouped_data_for_pool)
    print(f"Phase 2 complete: {len(partial_timelines)} partial timelines created.")
    print("\nPhase 3: Final merge of partial timelines...")
    final_audio = AudioSegment.silent(duration=max_end_time)
    for partial_audio in partial_timelines:
        final_audio = final_audio.overlay(partial_audio)
    print("Phase 3 complete.")
    return final_audio

def load_channel_file(filepath):
    if not os.path.isfile(filepath):
        return None, os.path.basename(filepath)
    return AudioSegment.from_wav(filepath), os.path.basename(filepath)

def get_channel_order(layout_string):
    if layout_string == "5.1(side)":
        return ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Párhuzamosított szkript a szinkronhangok és a háttérzene összefűzésére."
    )
    parser.add_argument('project_name', help='A feldolgozandó projekt könyvtár neve a "workdir"-en belül.')
    args = parser.parse_args()
    project_name = args.project_name

    # Config és útvonalak beállítása... (változatlan)
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Hiba: A config.json fájl nem található itt: {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Hiba: A config.json fájl hibás formátumú.")
        return
    
    try:
        workdir_base = config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        input_subdir = subdirs['translated_splits']
        output_subdir = subdirs['film_dubbing']
        background_subdir = subdirs['separated_audio_background']
        extracted_audio_subdir = subdirs['extracted_audio']
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        return

    project_path = os.path.join(workdir_base, project_name)
    input_folder = os.path.join(project_path, input_subdir)
    output_dir = os.path.join(project_path, output_subdir)
    background_dir = os.path.join(project_path, background_subdir)
    extracted_audio_dir = os.path.join(project_path, extracted_audio_subdir)
    audio_info_path = os.path.join(extracted_audio_dir, 'original_audio_info.json')

    try:
        with open(audio_info_path, 'r', encoding='utf-8') as f:
            original_audio_info = json.load(f)
        print(f"Successfully loaded original audio info from: {audio_info_path}")
    except FileNotFoundError:
        original_audio_info = None
    except json.JSONDecodeError:
        print(f"Error: Corrupted original audio info file ('{audio_info_path}').")
        return

    if not os.path.isdir(input_folder):
        print(f"Error: Input directory '{input_folder}' does not exist.")
        return
    os.makedirs(output_dir, exist_ok=True)

    dubbed_timeline = create_dubbed_timeline_fully_parallel(input_folder)
    if not dubbed_timeline: return
    print("\nTranslated audio timeline created successfully.")
    
    background_file = next((os.path.join(background_dir, f) for f in sorted(os.listdir(background_dir)) if f.lower().endswith(('.wav', '.mp3', '.aac', '.flac', 'm4a', 'ogg'))), None) if os.path.isdir(background_dir) else None
    
    final_center_channel = dubbed_timeline
    if background_file:
        print(f"Adding background music from: {background_file}")
        try:
            background = AudioSegment.from_file(background_file)
            final_duration = len(dubbed_timeline)
            if len(background) < final_duration: background = background * (final_duration // len(background) + 1)
            background = background[:final_duration]
            final_center_channel = background.overlay(dubbed_timeline)
            print("Background music added.")
        except Exception as e:
            print(f"Error processing background music: {e}. Continuing without it.")
    else:
        print("No background music file found. Continuing without it.")

    final_audio = None
    if original_audio_info and original_audio_info.get('channels', 1) > 2:
        print(f"\nRebuilding {original_audio_info['channels']}-channel audio.")
        channel_files_info = original_audio_info.get('extracted_channel_files')
        channel_layout = original_audio_info.get('channel_layout')
        order = get_channel_order(channel_layout) if channel_layout else None
        
        if channel_files_info and order:
            channel_paths_to_load = [os.path.join(extracted_audio_dir, fname) for ch, fname in channel_files_info.items() if ch != 'FC']
            
            try:
                print(f"Loading {len(channel_paths_to_load)} original channels in parallel...")
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    loaded_channels = pool.map(load_channel_file, channel_paths_to_load)

                channel_segments = {os.path.splitext(fname)[0].split('_')[-1]: seg for seg, fname in loaded_channels if seg}
                channel_segments['FC'] = final_center_channel.set_channels(1)

                # --- JAVÍTOTT RÉSZ KEZDETE ---
                # 1. Minta-tulajdonságok meghatározása
                template_segment = next((channel_segments[ch] for ch in order if ch in channel_segments and ch != 'FC'), None)
                if not template_segment:
                    raise ValueError("Could not load any original channels to use as a template.")
                
                master_duration = len(template_segment)
                master_frame_rate = template_segment.frame_rate
                master_sample_width = template_segment.sample_width
                
                print(f"Master properties set from template: {master_duration} ms, {master_frame_rate} Hz, {master_sample_width*8}-bit.")

                # 2. Összes csatorna normalizálása és tulajdonságainak kényszerítése
                segments_in_order = []
                for ch_name in order:
                    segment = channel_segments.get(ch_name)
                    if not segment:
                        raise ValueError(f"Channel '{ch_name}' is missing.")
                    
                    # Hossz normalizálása
                    current_len = len(segment)
                    if current_len < master_duration:
                        padding = AudioSegment.silent(duration=master_duration - current_len, frame_rate=master_frame_rate)
                        segment += padding
                    elif current_len > master_duration:
                        segment = segment[:master_duration]
                    
                    # Tulajdonságok kényszerítése a tökéletes egyezésért
                    segment = segment.set_frame_rate(master_frame_rate).set_sample_width(master_sample_width)
                    segments_in_order.append(segment)
                # --- JAVÍTOTT RÉSZ VÉGE ---

                if len(segments_in_order) == len(order):
                    print("Combining all length- and property-normalized channels...")
                    final_audio = AudioSegment.from_mono_audiosegments(*segments_in_order)
                    print("Multichannel audio successfully combined.")
                else:
                    raise ValueError("One or more audio channels were missing after normalization.")

            except Exception as e:
                print(f"Hiba a többcsatornás hang újraépítése során: {e}")
                print("Exporting dubbed track as stereo as a fallback.")
                final_audio = final_center_channel
        else:
            print("Warning: Multichannel info is incomplete. Exporting as stereo.")
            final_audio = final_center_channel
    else:
        print("\nOriginal audio is stereo or mono. Exporting the final mix directly.")
        final_audio = final_center_channel

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_file = os.path.join(output_dir, f"{timestamp}.wav")
    print(f"\nExporting final audio to: {output_file}")
    
    try:
        final_audio.export(output_file, format="wav")
        print(f"A végső hangfájl sikeresen elmentve ide: '{output_file}'")
    except Exception as e:
        print(f"Hiba a végső fájl exportálásakor: {e}")

if __name__ == "__main__":
    main()