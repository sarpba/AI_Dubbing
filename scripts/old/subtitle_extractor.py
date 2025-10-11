import os
import subprocess
import argparse

def extract_subtitles(input_file, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use ffprobe to get subtitle stream information in CSV format (order preserved)
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 's',
        '-show_entries', 'stream=index:stream_tags=language',
        '-of', 'csv=p=0', input_file
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error extracting subtitle information from {input_file}: {result.stderr}")
        return

    # Each line is expected in the format: index,language
    lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    if not lines:
        print("No subtitle streams found.")
        return

    print("Subtitle info:", lines)  # Debug print

    subtitle_count = {}
    # Use enumerate to get ffmpeg's positional mapping (order preserved as in ffprobe output)
    for pos, line in enumerate(lines):
        parts = line.split(',')
        if len(parts) < 2:
            print(f"Skipping unrecognized line: {line}")
            continue
        # We ignore the global stream index and rely on the order (pos)
        language_code = parts[1].strip() or "und"  # fallback to "und" if language not specified

        # Determine the output file name; first occurrence gets language_code.srt, subsequent get suffix
        if language_code not in subtitle_count:
            subtitle_count[language_code] = 1
            output_file_name = f"{language_code}.srt"
        else:
            subtitle_count[language_code] += 1
            output_file_name = f"{language_code}_{subtitle_count[language_code]}.srt"

        output_path = os.path.join(output_dir, output_file_name)

        extract_command = [
            'ffmpeg', '-i', input_file, '-map', f'0:s:{pos}?',
            '-c:s', 'srt', output_path, '-y'
        ]
        print(f"Extract command: {' '.join(extract_command)}")
        extract_result = subprocess.run(extract_command, capture_output=True, text=True)

        if extract_result.returncode != 0:
            print(f"Error extracting subtitle stream at position {pos} from {input_file}: {extract_result.stderr}")
        else:
            print(f"Subtitle extracted to {output_path}")

        # Verify the output file exists and is non-empty
        if os.path.exists(output_path):
            if os.path.getsize(output_path) > 0:
                print(f"Output file {output_path} exists and is not empty.")
            else:
                print(f"Output file {output_path} exists but is empty.")
        else:
            print(f"Output file {output_path} does not exist or is empty.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract subtitles from an MKV file.")
    parser.add_argument('-i', '--input', required=True, help='Input MKV file')
    parser.add_argument('-o', '--output', required=True, help='Output directory for subtitle files')

    args = parser.parse_args()
    extract_subtitles(args.input, args.output)
