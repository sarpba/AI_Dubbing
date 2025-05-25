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

def is_sentence_end(word_text):
    """
    Checks if a word ends with sentence-terminating punctuation.
    Considers common cases and avoids splitting on abbreviations like Mr. or U.S.
    """
    if not word_text:
        return False
    # Basic sentence terminators
    if word_text.endswith(('.', '?', '!')):
        # Avoid splitting on common abbreviations or initials
        if len(word_text) > 1 and word_text[-2].isalpha() and word_text[-2].isupper(): # e.g., U.S. or Mr.
            if len(word_text) == 2 and word_text[0].isupper(): # Single letter initial like "A."
                 return False # Typically not a sentence end in transcripts
            if re.match(r'\b([A-Za-z]\.){2,}\b', word_text): # Matches A.B.C.
                return False
            if re.match(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.$', word_text, re.IGNORECASE):
                return False
        # Check for numbers like 1. or 2. which might not be sentence ends in lists
        if word_text[-2].isdigit() and len(word_text) > 1 and word_text[:-1].replace('.', '').isdigit():
            return False # e.g. "1.", "2."
        return True
    return False

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

    # --- Simplified Segment-Based Chunking Logic ---
    all_errors = []
    chunk_counter = 0
    output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
    os.makedirs(output_subdir, exist_ok=True)
    default_speaker = "UNKNOWN"

    if "segments" not in json_content or not isinstance(json_content["segments"], list):
        logging.warning(f"No 'segments' list found in '{json_path}'. Cannot process.")
        return f"No 'segments' list found in '{json_path}'"

    for segment_index, segment_dict in enumerate(json_content["segments"]):
        if not isinstance(segment_dict, dict):
            logging.warning(f"Segment at index {segment_index} in {base_name} is not a dict. Skipping.")
            continue

        try:
            seg_start_time = float(segment_dict['start'])
            seg_end_time = float(segment_dict['end'])
            seg_text_full = segment_dict.get('text', "") 
            seg_speaker = segment_dict.get('speaker', default_speaker)
            if not seg_speaker: seg_speaker = default_speaker

            seg_words_raw = segment_dict.get('words', [])
            if not isinstance(seg_words_raw, list):
                logging.warning(f"Words for segment {segment_index} in {base_name} is not a list. Treating as empty for sub-chunking if segment is long.")
                seg_words_raw = []
            
            # Validate words if present
            seg_words = []
            if seg_words_raw:
                for word_idx, word_obj_raw in enumerate(seg_words_raw):
                    if isinstance(word_obj_raw, dict) and 'start' in word_obj_raw and 'end' in word_obj_raw and 'word' in word_obj_raw:
                        try:
                            # Ensure word times are float, relative to audio start
                            word_obj_validated = {
                                'start': float(word_obj_raw['start']),
                                'end': float(word_obj_raw['end']),
                                'word': str(word_obj_raw['word'])
                            }
                            seg_words.append(word_obj_validated)
                        except (ValueError, TypeError):
                            logging.warning(f"Invalid time format in word {word_idx} of segment {segment_index} in {base_name}. Skipping word: {word_obj_raw}")
                    else:
                        logging.warning(f"Invalid word object format in segment {segment_index} in {base_name}. Skipping word: {word_obj_raw}")


        except (KeyError, ValueError, TypeError) as e:
            logging.warning(f"Skipping segment at index {segment_index} in {base_name} due to missing/invalid essential keys (start, end, text, speaker): {e}")
            continue

        seg_duration = seg_end_time - seg_start_time

        if seg_duration <= 0.001: # Using a small epsilon for duration check
            logging.debug(f"Skipping segment {segment_index} with zero or negligible duration {seg_duration:.3f}s in {base_name}")
            continue

        if seg_duration <= max_chunk_duration:
            logging.debug(f"Exporting segment {segment_index} as a single chunk (Speaker: {seg_speaker}, Duration: {seg_duration:.3f}s)")
            export_errors, success = export_chunk(audio, seg_start_time, seg_end_time, seg_text_full, seg_speaker, output_subdir, export_format, export_extension, base_name, min_duration)
            all_errors.extend(export_errors)
            if success:
                chunk_counter += 1
        else:
            # Segment is too long, needs to be split
            num_sub_chunks = math.ceil(seg_duration / max_chunk_duration)
            logging.info(f"Segment {segment_index} (Speaker: {seg_speaker}, Duration: {seg_duration:.3f}s) is too long. Splitting into {num_sub_chunks} sub-chunks based on max_duration {max_chunk_duration}s.")

            if seg_words:
                current_word_cursor = 0
                for k in range(num_sub_chunks):
                    # Determine the time boundaries for this sub-chunk's words
                    # This aims for roughly equal time distribution among sub-chunks
                    chunk_ideal_start_time_abs = seg_start_time + k * (seg_duration / num_sub_chunks)
                    chunk_ideal_end_time_abs = seg_start_time + (k + 1) * (seg_duration / num_sub_chunks)
                    
                    if k == num_sub_chunks - 1: # Last sub-chunk should go to the segment's end
                        chunk_ideal_end_time_abs = seg_end_time

                    sub_chunk_word_objects_for_this_split = []
                    
                    # Collect words for this sub-chunk
                    temp_idx = current_word_cursor
                    first_word_start_time = -1
                    last_word_end_time = -1

                    while temp_idx < len(seg_words):
                        word = seg_words[temp_idx]
                        word_start_abs = word['start']
                        word_end_abs = word['end']

                        # Condition to include word:
                        # Word must start before the ideal end of this sub-chunk.
                        # For the last sub-chunk, all remaining words are included.
                        if k < num_sub_chunks - 1: # Not the last sub-chunk
                            if word_start_abs >= chunk_ideal_end_time_abs: 
                                break # This word belongs to the next sub-chunk
                        
                        if not sub_chunk_word_objects_for_this_split: # First word for this sub-chunk
                            first_word_start_time = word_start_abs
                        
                        sub_chunk_word_objects_for_this_split.append(word)
                        last_word_end_time = word_end_abs
                        temp_idx += 1
                        
                        # If not the last sub-chunk, and this word's *end* crosses the ideal boundary,
                        # it's the last word for this sub-chunk.
                        if k < num_sub_chunks - 1 and word_end_abs >= chunk_ideal_end_time_abs:
                            break
                    
                    current_word_cursor = temp_idx # Update main cursor

                    if sub_chunk_word_objects_for_this_split:
                        # Use actual start/end times from the collected words
                        actual_sub_chunk_start = sub_chunk_word_objects_for_this_split[0]['start']
                        actual_sub_chunk_end = sub_chunk_word_objects_for_this_split[-1]['end']
                        sub_chunk_text = " ".join([w['word'] for w in sub_chunk_word_objects_for_this_split]).strip()

                        if actual_sub_chunk_start >= actual_sub_chunk_end and not (actual_sub_chunk_start == actual_sub_chunk_end == 0): # Allow 0,0 for safety if it happens
                             logging.warning(f"Sub-chunk {k+1}/{num_sub_chunks} from segment {segment_index} (words) has invalid times {actual_sub_chunk_start:.3f}>={actual_sub_chunk_end:.3f}. Skipping.")
                             continue
                        
                        logging.debug(f"  Exporting sub-chunk {k+1}/{num_sub_chunks} (Words-based, Speaker: {seg_speaker}): {actual_sub_chunk_start:.3f}-{actual_sub_chunk_end:.3f}")
                        export_errors_sub, success_sub = export_chunk(audio, actual_sub_chunk_start, actual_sub_chunk_end, sub_chunk_text, seg_speaker, output_subdir, export_format, export_extension, base_name, min_duration)
                        all_errors.extend(export_errors_sub)
                        if success_sub:
                            chunk_counter += 1
                    elif k < num_sub_chunks -1 : # Only warn if not the last chunk and no words were found (might be due to sparse words)
                        logging.warning(f"Sub-chunk {k+1}/{num_sub_chunks} of segment {segment_index} (words) found no words. Ideal time: {chunk_ideal_start_time_abs:.3f}-{chunk_ideal_end_time_abs:.3f}")

            else: # No word-level details, split audio and text proportionally
                logging.warning(f"Segment {segment_index} in {base_name} (Speaker: {seg_speaker}) is too long but has no word details. Splitting audio and text proportionally.")
                for k in range(num_sub_chunks):
                    sub_s = seg_start_time + k * (seg_duration / num_sub_chunks)
                    sub_e = seg_start_time + (k + 1) * (seg_duration / num_sub_chunks)
                    sub_e = min(sub_e, seg_end_time) 

                    if sub_s >= sub_e and not (sub_s == sub_e == 0): # Allow 0,0 for safety
                        logging.warning(f"Sub-chunk {k+1}/{num_sub_chunks} from segment {segment_index} (proportional) has invalid times {sub_s:.3f}>={sub_e:.3f}. Skipping.")
                        continue
                    
                    text_len = len(seg_text_full)
                    text_s_char_idx = math.floor(text_len * (k / num_sub_chunks))
                    text_e_char_idx = math.floor(text_len * ((k + 1) / num_sub_chunks))
                    sub_text = seg_text_full[text_s_char_idx:text_e_char_idx].strip()
                    
                    if not sub_text and (sub_e - sub_s) > 0.05 : # If audio duration is non-trivial but text is empty
                        sub_text = f"[chunk_{k+1}_of_{num_sub_chunks}_no_text_assigned]"
                        logging.debug(f"  Sub-chunk {k+1}/{num_sub_chunks} (Proportional text) for segment {segment_index} resulted in empty text. Using placeholder. Audio: {sub_s:.3f}-{sub_e:.3f}")

                    logging.debug(f"  Exporting sub-chunk {k+1}/{num_sub_chunks} (Proportional, Speaker: {seg_speaker}): {sub_s:.3f}-{sub_e:.3f}")
                    export_errors_sub, success_sub = export_chunk(audio, sub_s, sub_e, sub_text, seg_speaker, output_subdir, export_format, export_extension, base_name, min_duration)
                    all_errors.extend(export_errors_sub)
                    if success_sub:
                        chunk_counter += 1
    
    # --- End of processing for this JSON file ---
    stats = f"File: '{base_name}' | Exported chunks: {chunk_counter}"
    if all_errors:
        logging.warning(f"Finished processing '{base_name}' with {len(all_errors)} errors/skipped chunks.")
        error_details = "\nErrors/Skipped:\n" + "\n".join(all_errors)
        return f"{stats}{error_details}"
    else:
        logging.info(f"Successfully processed '{base_name}'. Exported {chunk_counter} chunks.")
        return f"{stats}\nProcessing successful."

# Update process_directory signature
def process_directory(input_dir, output_dir, max_chunk_duration, min_duration):
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
        '--min_duration', '-m', type=float, default=0.0,
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
