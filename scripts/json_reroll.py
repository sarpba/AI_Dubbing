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
    return parser.parse_args()

def split_segment(segment, max_length):
    """
    Split a single segment into multiple segments if its duration exceeds max_length.
    Preferably split at punctuation marks.
    """
    splits = []
    words = segment['words']
    n = len(words)
    i = 0

    while i < n:
        sub_start = words[i]['start']
        sub_end = sub_start + max_length

        # Find the last word that ends before sub_end
        split_idx = -1
        for j in range(i, n):
            word_end = words[j]['end']
            if word_end > sub_end:
                break
            # Check if the word ends with punctuation
            if words[j]['word'].strip() and words[j]['word'].strip()[-1] in {',', '.', '?', '!', ';'}:
                split_idx = j

        if split_idx != -1 and split_idx >= i:
            # Split at split_idx
            sub_words = words[i:split_idx+1]
            new_segment = {
                'start': sub_words[0]['start'],
                'end': sub_words[-1]['end'],
                'text': reconstruct_text(sub_words),
                'words': sub_words,
                'speaker': segment.get('speaker')  # Use get to avoid KeyError
            }
            splits.append(new_segment)
            i = split_idx + 1
        else:
            # No suitable punctuation found, split at max_length or end of segment
            split_idx = i
            while split_idx < n and words[split_idx]['end'] <= sub_end:
                split_idx += 1
            if split_idx == i:
                # Single word exceeds max_length, force split after this word
                split_idx = i + 1
            sub_words = words[i:split_idx]
            new_segment = {
                'start': sub_words[0]['start'],
                'end': sub_words[-1]['end'],
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

def split_all_segments(segments, max_length):
    """
    Split all segments in the list based on max_length.
    """
    split_segments = []
    for segment in segments:
        segment_duration = segment['end'] - segment['start']
        if segment_duration <= max_length:
            split_segments.append(segment)
        else:
            split_segments.extend(split_segment(segment, max_length))
    return split_segments

def merge_segments(segments, distance, max_length):
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

        gap = seg['start'] - last['end']
        if seg_speaker == last_speaker and gap <= distance:
            # Calculate the potential new end time
            potential_end = seg['end']
            # Calculate the total duration if merged
            total_duration = potential_end - last['start']
            if total_duration <= max_length:
                # Merge the segments
                last['end'] = seg['end']
                last['text'] += ' ' + seg['text']
                last['words'].extend(seg['words'])
            else:
                # Cannot merge as it would exceed max_length
                merged.append(seg)
        else:
            merged.append(seg)
    return merged

def process_segments(segments, max_length, distance):
    """
    Process the segments by splitting and then merging them.
    """
    # Step 1: Split segments exceeding max_length
    split_segments = split_all_segments(segments, max_length)
    
    # Step 2: Sort the split segments by start time to ensure proper merging
    split_segments.sort(key=lambda x: x['start'])
    
    # Step 3: Merge segments based on the distance and speaker criteria without exceeding max_length
    merged_segments = merge_segments(split_segments, distance, max_length)
    
    return merged_segments

def main():
    args = parse_arguments()

    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read the input JSON file
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
    
    # Process the segments
    processed_segments = process_segments(original_segments, args.max_length, args.distance)
    
    # Update the data with processed segments
    data['segments'] = processed_segments
    
    # Remove 'word_segments' if it exists
    if 'word_segments' in data:
        del data['word_segments']
    
    # Write the output JSON file
    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print(f"Processed JSON has been saved to '{args.output}'.")
    except Exception as e:
        print(f"Error: Failed to write output file. {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
