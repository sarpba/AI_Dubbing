import os
import argparse
import json
import re
import math
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import timedelta
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(filename):
    """
    Sanitizes filenames by replacing forbidden characters.
    Replaces colons and commas with hyphens.
    """
    sanitized = filename.replace(":", "-").replace(",", "-")
    sanitized = re.sub(r'[\\/*?:"<>|]', '', sanitized)
    return sanitized

def format_timedelta_srt(seconds):
    """
    Formats seconds into SRT timestamp format: HH:MM:SS,mmm
    """
    if seconds < 0:
        seconds = 0
    # Ensure milliseconds are handled correctly, even for very small negative numbers close to zero
    if seconds < 0.001 and seconds > -0.000001:
         seconds = 0
    td = timedelta(seconds=abs(seconds)) # Use abs to handle potential tiny negative floats near zero
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    # Calculate milliseconds from the fractional part of the original seconds value
    milliseconds = int((abs(seconds) - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

# Add min_duration parameter to export_chunk
def export_chunk(audio, start_time, end_time, text, speaker, output_subdir, export_format, export_extension, base_name, min_duration):
    """Helper function to export a single audio chunk and its text file."""
    errors = []
    chunk_duration = end_time - start_time

    # Check minimum duration
    if chunk_duration < min_duration:
        logging.debug(f"Skipping chunk shorter than min_duration ({min_duration}s): Duration={chunk_duration:.3f}s, Text='{text[:50]}...'")
        return errors, False # Indicate failure (skipped)

    try:
        if start_time >= end_time:
             logging.warning(f"Skipping invalid chunk slice: Start={start_time:.3f} >= End={end_time:.3f} in '{base_name}'")
             return errors, False # Indicate failure

        chunk_start_ms = int(start_time * 1000)
        chunk_end_ms = int(end_time * 1000)

        # Ensure slice indices are within bounds and valid
        chunk_start_ms = max(0, chunk_start_ms)
        chunk_end_ms = min(len(audio), chunk_end_ms)

        if chunk_start_ms >= chunk_end_ms:
            logging.warning(f"Skipping invalid chunk slice after bounds check: Start={chunk_start_ms}ms >= End={chunk_end_ms}ms in '{base_name}'")
            return errors, False # Indicate failure

        audio_chunk = audio[chunk_start_ms:chunk_end_ms]

        start_timestamp_str = format_timedelta_srt(start_time)
        end_timestamp_str = format_timedelta_srt(end_time)
        filename_base = f"{sanitize_filename(start_timestamp_str)}_{sanitize_filename(end_timestamp_str)}_{sanitize_filename(speaker)}"

        output_audio_path = os.path.join(output_subdir, f"{filename_base}.{export_extension}")
        output_text_path = os.path.join(output_subdir, f"{filename_base}.txt")

        audio_chunk.export(output_audio_path, format=export_format)
        with open(output_text_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        logging.debug(f"Exported chunk: {filename_base} ({start_time:.3f}s - {end_time:.3f}s)")
        return errors, True # Indicate success

    except Exception as e:
        error_msg = f"Error processing chunk ({start_time:.3f}-{end_time:.3f}) in '{base_name}': {e}"
        logging.error(error_msg, exc_info=True)
        errors.append(error_msg)
        return errors, False # Indicate failure

# Update process_json_file signature to accept min_duration
def process_json_file(args):
    json_path, audio_dir, output_dir, relative_path, max_chunk_duration, min_duration = args # Added min_duration
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    logging.info(f"Processing JSON: {json_path}")

    # --- Audio and JSON Loading (same as before) ---
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus']
    audio_file = None
    original_extension = None
    for ext in audio_extensions:
        potential_audio_path = os.path.join(audio_dir, base_name + ext)
        if os.path.exists(potential_audio_path):
            audio_file = potential_audio_path
            original_extension = ext.lstrip('.').lower()
            logging.info(f"Found audio file: {audio_file}")
            break
    if not audio_file:
        logging.warning(f"Audio file not found for base name '{base_name}' in directory '{audio_dir}'. Skipping JSON: '{json_path}'")
        return f"Audio file not found for '{base_name}' (JSON: '{json_path}')"
    try:
        logging.debug(f"Loading audio: {audio_file}")
        audio = AudioSegment.from_file(audio_file)
        export_extension = original_extension if original_extension in ['wav', 'mp3', 'ogg', 'flac'] else 'wav'
        export_format = export_extension
        logging.debug(f"Audio loaded successfully. Export format set to: {export_format}")
    except Exception as e:
        logging.error(f"Error loading audio file '{audio_file}': {e}", exc_info=True)
        return f"Error loading audio file '{audio_file}': {e}"
    try:
        logging.debug(f"Loading JSON: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        logging.debug(f"JSON loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading JSON file '{json_path}': {e}", exc_info=True)
        return f"Error loading JSON file '{json_path}': {e}"

    # --- Prepare word list ---
    words_to_process = []
    default_speaker = "UNKNOWN"
    if "word_segments" in json_content and isinstance(json_content["word_segments"], list) and json_content["word_segments"]:
        logging.debug("Using 'word_segments' as the primary source.")
        words_to_process = json_content["word_segments"]
        for i, word_dict in enumerate(words_to_process):
            if isinstance(word_dict, dict):
                 word_dict['speaker'] = default_speaker
                 word_dict['original_index'] = i # Keep track of original position
            else:
                 logging.warning(f"Unexpected item format in word_segments: {word_dict}. Skipping.")
        words_to_process = [word for word in words_to_process if isinstance(word, dict)]
    elif "segments" in json_content and isinstance(json_content["segments"], list):
        logging.debug("Using 'segments' containing 'words'.")
        flat_word_list = []
        original_idx_counter = 0
        loaded_segments = json_content.get("segments", [])
        for segment_dict in loaded_segments:
            if not isinstance(segment_dict, dict): continue
            segment_words = segment_dict.get("words", [])
            speaker = segment_dict.get("speaker", default_speaker)
            for word_dict in segment_words:
                if isinstance(word_dict, dict):
                    word_dict['speaker'] = speaker
                    word_dict['original_index'] = original_idx_counter
                    flat_word_list.append(word_dict)
                    original_idx_counter += 1
                else:
                    logging.warning(f"Unexpected word format in segment: {word_dict}. Skipping.")
        words_to_process = flat_word_list
    else:
        logging.warning(f"No 'word_segments' or 'segments' with 'words' found in '{json_path}'. Cannot process.")
        return f"No processable word data found in '{json_path}'"

    if not words_to_process:
        logging.warning(f"'{json_path}' contains segment structures but no actual words. Skipping.")
        return f"No words found to process in '{json_path}'"

    # --- New Time-Based Chunking Logic ---
    chunk_counter = 0
    all_errors = [] # Renamed from errors to avoid conflict
    output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
    os.makedirs(output_subdir, exist_ok=True)

    current_chunk_words = []
    max_inter_word_gap = 1.0 # Maximum allowed gap in seconds before forcing a split

    for i, current_word in enumerate(words_to_process):
        # Validate current word data
        if not isinstance(current_word, dict) or 'start' not in current_word or 'end' not in current_word or 'word' not in current_word:
            logging.warning(f"Skipping invalid word data at index {i}: {current_word}")
            continue
        
        # Ensure times are floats
        try:
            current_word['start'] = float(current_word['start'])
            current_word['end'] = float(current_word['end'])
        except (ValueError, TypeError):
             logging.warning(f"Skipping word with invalid time format at index {i}: {current_word}")
             continue

        # Calculate gap to previous word
        gap_to_previous = 0.0
        force_split_due_to_gap = False
        if i > 0:
            previous_word = words_to_process[i-1]
            # Ensure previous word also has valid data, especially 'end' time
            if isinstance(previous_word, dict) and 'end' in previous_word:
                 try:
                     previous_end_time = float(previous_word['end'])
                     gap_to_previous = current_word['start'] - previous_end_time
                     if gap_to_previous > max_inter_word_gap:
                         force_split_due_to_gap = True
                         logging.debug(f"  Gap > {max_inter_word_gap}s ({gap_to_previous:.3f}s) before word '{current_word['word']}' (idx {i}). Forcing split.")
                 except (ValueError, TypeError):
                      logging.warning(f"Could not calculate gap before word index {i} due to invalid previous word end time: {previous_word.get('end')}")
            else:
                 logging.warning(f"Could not calculate gap before word index {i} due to invalid previous word data: {previous_word}")


        # Calculate potential duration if current word is added
        potential_chunk_duration = 0.0
        force_split_due_to_duration = False
        if current_chunk_words:
            potential_start_time = current_chunk_words[0]['start']
            potential_end_time = current_word['end']
            potential_chunk_duration = potential_end_time - potential_start_time
            if potential_chunk_duration > max_chunk_duration:
                force_split_due_to_duration = True
                logging.debug(f"  Potential duration > {max_chunk_duration}s ({potential_chunk_duration:.3f}s) if adding word '{current_word['word']}' (idx {i}). Forcing split.")
        else:
            # If chunk is empty, potential duration is just the current word's duration
            potential_chunk_duration = current_word['end'] - current_word['start']
            # A single word exceeding max_duration is unlikely but possible
            if potential_chunk_duration > max_chunk_duration:
                 logging.warning(f"Single word '{current_word['word']}' (idx {i}) duration ({potential_chunk_duration:.3f}s) exceeds max_duration ({max_chunk_duration}s). It will form its own chunk.")
                 # We don't set force_split_due_to_duration=True here, as it will be handled when the word is added


        # --- Finalize the PREVIOUS chunk if needed ---
        if current_chunk_words and (force_split_due_to_gap or force_split_due_to_duration):
            chunk_start_time = current_chunk_words[0]['start']
            chunk_end_time = current_chunk_words[-1]['end']
            chunk_text = " ".join([w.get("word", "") for w in current_chunk_words]).strip()
            chunk_speaker = current_chunk_words[0].get('speaker', default_speaker) # Use first word's speaker

            logging.debug(f"Finalizing chunk: Words {current_chunk_words[0]['original_index']}-{current_chunk_words[-1]['original_index']}, Time {chunk_start_time:.3f}-{chunk_end_time:.3f}")
            export_errors, success = export_chunk(audio, chunk_start_time, chunk_end_time, chunk_text, chunk_speaker, output_subdir, export_format, export_extension, base_name, min_duration)
            all_errors.extend(export_errors)
            if success:
                chunk_counter += 1

            # Start new chunk
            current_chunk_words = []

        # --- Add current word to the (potentially new) chunk ---
        # Check if the word itself is valid before adding
        if current_word['end'] > current_word['start']:
             current_chunk_words.append(current_word)
        else:
             logging.warning(f"Skipping word with invalid duration (end <= start) at index {i}: {current_word}")


    # --- Process the last remaining chunk after the loop ---
    if current_chunk_words:
        chunk_start_time = current_chunk_words[0]['start']
        chunk_end_time = current_chunk_words[-1]['end']
        chunk_text = " ".join([w.get("word", "") for w in current_chunk_words]).strip()
        chunk_speaker = current_chunk_words[0].get('speaker', default_speaker)

        logging.debug(f"Finalizing last chunk: Words {current_chunk_words[0]['original_index']}-{current_chunk_words[-1]['original_index']}, Time {chunk_start_time:.3f}-{chunk_end_time:.3f}")
        export_errors, success = export_chunk(audio, chunk_start_time, chunk_end_time, chunk_text, chunk_speaker, output_subdir, export_format, export_extension, base_name, min_duration)
        all_errors.extend(export_errors)
        if success:
            chunk_counter += 1


    # --- End of processing for this JSON file ---
    stats = f"File: '{base_name}' | Exported chunks: {chunk_counter}" # Changed 'Processed' to 'Exported'
    if all_errors:
        logging.warning(f"Finished processing '{base_name}' with {len(all_errors)} errors/skipped chunks.")
        # Combine stats and errors for return
        error_details = "\nErrors/Skipped:\n" + "\n".join(all_errors)
        return f"{stats}{error_details}"
    else:
        logging.info(f"Successfully processed '{base_name}'. Exported {chunk_counter} chunks.")
        return f"{stats}\nProcessing successful."


# Update process_directory signature
def process_directory(input_dir, output_dir, max_chunk_duration, min_duration): # Added min_duration
    json_files_args = []
    logging.info(f"Scanning input directory: {input_dir}")
    for root, dirs, files in os.walk(input_dir):
        # Skip hidden directories like .ipynb_checkpoints
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                audio_dir_for_json = root # Assume audio is in the same dir as json
                relative_path = os.path.relpath(json_path, input_dir)
                # Add min_duration to the arguments tuple
                json_files_args.append((json_path, audio_dir_for_json, output_dir, relative_path, max_chunk_duration, min_duration))
                logging.debug(f"Found JSON: {json_path}, Relative path: {relative_path}")

    total_files = len(json_files_args)
    if total_files == 0:
        logging.warning(f"No JSON files found in '{input_dir}'.")
        print("No JSON files found to process.")
        return

    logging.info(f"Found {total_files} JSON files to process.")
    results = []
    # Consider using max_workers=os.cpu_count() or slightly less if memory is an issue
    with ProcessPoolExecutor() as executor:
        future_to_path = {executor.submit(process_json_file, args): args[0] for args in json_files_args}
        for future in tqdm(as_completed(future_to_path), total=total_files, desc="Processing Files"):
            json_path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                # Optionally print results immediately or collect and print later
                # print(result)
            except Exception as exc:
                error_msg = f"Critical error processing '{json_path}': {exc}"
                logging.critical(error_msg, exc_info=True)
                results.append(error_msg)
                # print(error_msg)

    print("\n--- Processing Summary ---")
    success_count = 0
    error_count = 0
    for res in results:
        print(res)
        if "Error" in res or "error" in res or "failed" in res or "not found" in res:
             error_count +=1
        elif "successful" in res or "Processed chunks" in res:
             success_count +=1
    print("-------------------------")
    print(f"Total files processed: {total_files}")
    print(f"Successful: {success_count}")
    print(f"Files with errors/warnings: {error_count}")
    print("-------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Splits audio files into chunks based on word-level timestamps from JSON transcriptions.",
        epilog="Example:\n  python scripts/splitter.py -i ./input_data/ -o ./output_chunks/ --max_duration 12",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input_dir', '-i', type=str, required=True,
        help='Input directory containing JSON files and corresponding audio files (in the same subdirectories).'
    )

    parser.add_argument(
        '--output_dir', '-o', type=str, required=True,
        help='Output directory where the audio chunks and text files will be saved, preserving relative structure.'
    )

    parser.add_argument(
        '--max_duration', '-d', type=float, default=10.0,
        help='Maximum duration (in seconds) for each audio chunk. Default: 12.0'
    )

    parser.add_argument(
        '--min_duration', '-m', type=float, default=3.0,
        help='Minimum duration (in seconds) for each audio chunk. Shorter chunks will be skipped. Default: 5.0'
    )

    parser.add_argument(
        '--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO',
        help='Set the logging level. Default: INFO'
    )

    args = parser.parse_args()

    # Update logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Maximum chunk duration: {args.max_duration}s")
    logging.info(f"Minimum chunk duration: {args.min_duration}s") # Log the new arg
    logging.info(f"Log level: {args.log_level}")


    process_directory(args.input_dir, args.output_dir, args.max_duration, args.min_duration) # Pass min_duration

if __name__ == "__main__":
    main()
