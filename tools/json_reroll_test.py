import argparse
import json
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Rearrange JSON segments based on max_length and distance.")
    parser.add_argument('-i', '--input', required=True, help='Input JSON file path.')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file path.')
    parser.add_argument('--max_length', type=float, default=10.0, help='Maximum segment length in seconds (default: 10).')
    parser.add_argument('-d', '--distance', type=float, default=1.0, help='Maximum allowed distance between segments in seconds for merging (default: 1).')
    parser.add_argument('--just_split', action='store_true', help='Only split long segments without merging short ones.')
    return parser.parse_args()

def split_segment(segment, max_length, missing_keys):
    """
    Split a single segment into multiple segments if its duration exceeds max_length.
    Preferably split at punctuation marks.
    """
    splits = []
    words = segment.get('words', [])
    n = len(words)
    i = 0

    while i < n:
        # Ellenőrizzük, hogy a 'start' kulcs létezik-e
        sub_start = words[i].get('start')
        if sub_start is None:
            missing_keys.append({'segment_start': segment.get('start'), 'word': words[i], 'missing': 'start'})
            i += 1
            continue

        sub_end = sub_start + max_length

        # Megkeressük az utolsó szót, amely az 'sub_end' előtt végződik
        split_idx = -1
        for j in range(i, n):
            word_end = words[j].get('end')
            if word_end is None:
                missing_keys.append({'segment_start': segment.get('start'), 'word': words[j], 'missing': 'end'})
                break
            if word_end > sub_end:
                break
            # Ellenőrizzük, hogy a szó írásjelre végződik-e
            if words[j]['word'].strip() and words[j]['word'].strip()[-1] in {',', '.', '?', '!', ';'}:
                split_idx = j

        if split_idx != -1 and split_idx >= i:
            # Darabolás a split_idx-nél
            sub_words = words[i:split_idx+1]
            new_segment = {
                'start': sub_words[0].get('start', sub_start),
                'end': sub_words[-1].get('end', sub_start),
                'text': reconstruct_text(sub_words),
                'words': sub_words,
                'speaker': segment.get('speaker')  # Use get to avoid KeyError
            }
            splits.append(new_segment)
            i = split_idx + 1
        else:
            # Nincs megfelelő írásjel, így darabolás max_length-nél vagy a szegmens végénél
            split_idx = i
            while split_idx < n and words[split_idx].get('end', 0) <= sub_end:
                split_idx += 1
            if split_idx == i:
                # Egyetlen szó is meghaladja a max_length-et, így kényszerített darabolás
                split_idx = i + 1
            sub_words = words[i:split_idx]
            # Ellenőrizzük, hogy az 'end' kulcs létezik-e
            end_time = sub_words[-1].get('end')
            if end_time is None:
                missing_keys.append({'segment_start': segment.get('start'), 'word': sub_words[-1], 'missing': 'end'})
                end_time = sub_words[-1].get('start', 0)  # Alapértelmezett érték
            new_segment = {
                'start': sub_words[0].get('start', sub_start),
                'end': end_time,
                'text': reconstruct_text(sub_words),
                'words': sub_words,
                'speaker': segment.get('speaker')  # Use get to avoid KeyError
            }
            splits.append(new_segment)
            i = split_idx

    return splits

def reconstruct_text(words):
    """
    Reconstruct the text from a list of word dictionaries.
    Handles spacing around punctuation appropriately.
    """
    text = ""
    for idx, word_dict in enumerate(words):
        word = word_dict['word']
        if idx > 0:
            # Determine if space is needed
            prev_word = words[idx - 1]['word']
            if not prev_word.endswith((',', '.', '?', '!', ';')):
                text += ' '
        text += word
    return text

def split_all_segments(segments, max_length, missing_keys):
    """
    Split all segments in the list based on max_length.
    """
    split_segments = []
    for segment in segments:
        segment_duration = segment.get('end', 0) - segment.get('start', 0)
        if segment_duration <= max_length:
            split_segments.append(segment)
        else:
            split_segments.extend(split_segment(segment, max_length, missing_keys))
    return split_segments

def merge_segments(segments, distance, max_length, missing_keys):
    """
    Merge consecutive segments that have the same speaker and are within the specified distance.
    Ensure that the merged segment does not exceed max_length.
    Segments missing 'speaker' are not merged.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        # Check if 'speaker' exists in both segments
        seg_speaker = seg.get('speaker')
        last_speaker = last.get('speaker')

        if seg_speaker is None or last_speaker is None:
            # Cannot merge segments without 'speaker' information
            merged.append(seg)
            continue

        gap = seg.get('start', 0) - last.get('end', 0)
        if seg_speaker == last_speaker and gap <= distance:
            # Calculate the potential new end time
            potential_end = seg.get('end', last.get('end', 0))
            # Calculate the total duration if merged
            total_duration = potential_end - last.get('start', 0)
            if total_duration <= max_length:
                # Merge the segments
                last['end'] = seg.get('end', last.get('end', 0))
                last['text'] += ' ' + seg.get('text', '')
                last['words'].extend(seg.get('words', []))
            else:
                # Cannot merge as it would exceed max_length
                merged.append(seg)
        else:
            merged.append(seg)
    return merged

def process_segments(segments, max_length, distance, just_split, missing_keys):
    """
    Process the segments by splitting and optionally merging them.
    """
    # Step 1: Split segments exceeding max_length
    split_segments = split_all_segments(segments, max_length, missing_keys)
    
    if just_split:
        # If only splitting is requested, skip merging
        return split_segments
    
    # Step 2: Sort the split segments by start time to ensure proper merging
    split_segments.sort(key=lambda x: x.get('start', 0))
    
    # Step 3: Merge segments based on the distance and speaker criteria without exceeding max_length
    merged_segments = merge_segments(split_segments, distance, max_length, missing_keys)
    
    return merged_segments

def main():
    args = parse_arguments()

    # Ellenőrizd, hogy a bemeneti fájl létezik
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Olvasd be a bemeneti JSON fájlt
    try:
        with open(args.input, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file. {e}", file=sys.stderr)
        sys.exit(1)

    if 'segments' not in data:
        print("Error: JSON does not contain 'segments' field.", file=sys.stderr)
        sys.exit(1)

    original_segments = data['segments']
    
    # Lista a hiányzó kulcsok gyűjtésére
    missing_keys = []
    
    # Feldolgozd a szegmenseket
    processed_segments = process_segments(original_segments, args.max_length, args.distance, args.just_split, missing_keys)
    
    # Frissítsd az adatokat a feldolgozott szegmensekkel
    data['segments'] = processed_segments
    
    # Távolítsd el a 'word_segments' kulcsot, ha létezik
    if 'word_segments' in data:
        del data['word_segments']
    
    # Írd ki a kimeneti JSON fájlt
    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print(f"Processed JSON has been saved to '{args.output}'.")
    except Exception as e:
        print(f"Error: Failed to write output file. {e}", file=sys.stderr)
        sys.exit(1)
    
    # Ha vannak hiányzó kulcsok, listázd ki őket
    if missing_keys:
        print("\nFigyelmeztetések:")
        for idx, issue in enumerate(missing_keys, 1):
            segment_start = issue.get('segment_start', 'ismeretlen')
            word = issue.get('word', {})
            missing = issue.get('missing', 'ismeretlen')
            print(f"{idx}. Segment start: {segment_start}, Word: {word}, Missing key: '{missing}'")
    else:
        print("Nincs hiányzó 'start' vagy 'end' kulcs a szótárakban.")

if __name__ == "__main__":
    main()
