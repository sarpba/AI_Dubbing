import os
import subprocess
from multiprocessing import Pool
import argparse
import sys

def get_num_gpus():
    """Retrieves the number of available GPUs."""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, check=True, text=True)
        gpus = result.stdout.strip().split('\n')
        return len(gpus)
    except Exception as e:
        print("Failed to determine the number of GPUs. Defaulting to 1 GPU.")
        return 1

def validate_args(args):
    """Validates the command-line arguments based on the specified conditions."""
    # Mandatory -m argument
    if args.model_type not in ['F5-TTS', 'E2-TTS']:
        print("Error: The -m parameter must be either 'F5-TTS' or 'E2-TTS'.")
        sys.exit(1)
    
    # Mandatory one of -r or -rd
    if not args.ref_audio and not args.ref_audio_dir:
        print("Error: Either -r or -rd parameter must be provided.")
        sys.exit(1)
    
    # If -rd is provided, -rt is also mandatory
    if args.ref_audio_dir and not args.ref_text_dir:
        print("Error: When using -rd, the -rt REF_TEXT_DIR parameter must also be provided.")
        sys.exit(1)
    
    # If -rd is not provided, -s is mandatory
    if not args.ref_audio_dir and not args.ref_text:
        print("Error: Since -rd is not provided, the -s REFERENCE_TEXT parameter is mandatory.")
        sys.exit(1)
    
    # Mandatory one of -t, -f, or -rt_gen for generation
    if not args.gen_text and not args.gen_text_file and not args.gen_text_dir:
        print("Error: One of -t, -f, or -rt_gen parameters must be provided for generation.")
        sys.exit(1)
    
    # If -gen_text_dir is provided, -ref_audio_dir must also be provided
    if args.gen_text_dir and not args.ref_audio_dir:
        print("Error: When using -rt_gen REF_TEXT_DIR, the -rd REF_AUDIO_DIR parameter must also be provided.")
        sys.exit(1)
    
    # Check the checkpoint file extension
    if args.checkpoint and not args.checkpoint.endswith('.pt'):
        print("Error: The -p CHECKPOINT file must have a .pt extension.")
        sys.exit(1)
    
    # Validate the speed value
    if args.speed and not (0.1 <= args.speed <= 2.0):
        print("Error: The --speed parameter must be between 0.1 and 2.0.")
        sys.exit(1)

def construct_command(args, task_info, unique_output_subdir):
    """Constructs the f5-tts_infer-cli command based on the provided parameters."""
    command = ['f5-tts_infer-cli']
    
    # Add optional config file
    if args.config:
        command.extend(['-c', args.config])
    
    # Add mandatory model type
    command.extend(['-m', args.model_type])
    
    # Add optional checkpoint file
    if args.checkpoint:
        command.extend(['-p', args.checkpoint])
    
    # Add optional vocab file
    if args.vocab:
        command.extend(['-v', args.vocab])
    
    # Add reference audio
    if task_info['ref_audio']:
        command.extend(['-r', task_info['ref_audio']])
    
    # Add reference text
    if task_info['ref_text']:
        command.extend(['-s', task_info['ref_text']])
    
    # Add generated text
    if task_info['gen_text']:
        command.extend(['-t', task_info['gen_text']])
    elif task_info['gen_text_file']:
        command.extend(['-f', task_info['gen_text_file']])
    # Note: When using gen_text_dir, individual tasks handle their own gen_text
    
    # Add output directory
    command.extend(['-o', unique_output_subdir])
    
    # Add optional switches
    if args.remove_silence:
        command.append('--remove_silence')
    if args.load_vocoder_from_local:
        command.append('--load_vocoder_from_local')
    if args.speed:
        command.extend(['--speed', str(args.speed)])
    
    return command

def process_task(args, task, gpu_id):
    """Processes a single task using the specified GPU."""
    basename = os.path.splitext(os.path.basename(task['ref_audio']))[0]
    output_wav = os.path.join(args.output_dir, f"{basename}.wav")
    
    if os.path.exists(output_wav):
        print(f"The file has already been processed and exists: {output_wav}. Skipping.")
        return
    
    # Create a unique subdirectory for the output file
    unique_output_subdir = os.path.join(args.output_dir, f"{basename}_temp")
    os.makedirs(unique_output_subdir, exist_ok=True)
    
    # Determine the generated text
    if args.gen_text_dir:
        gen_text_path = os.path.join(args.gen_text_dir, f"{basename}.txt")
        if not os.path.isfile(gen_text_path):
            print(f"Error: Generated text file does not exist: {gen_text_path}. Skipping.")
            return
        with open(gen_text_path, 'r', encoding='utf-8') as f:
            gen_text = f.read().strip().lower()           
        task['gen_text'] = "... " + gen_text
    elif args.gen_text_file:
        task['gen_text'] = None  # Will be handled via '-f' flag
    # Else, gen_text is already provided via '-t'
    
    command = construct_command(args, task, unique_output_subdir)
    
    # Set the CUDA_VISIBLE_DEVICES environment variable to use the appropriate GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        print(f"Running: {' '.join(command)} (GPU {gpu_id})")
        subprocess.run(command, check=True, env=env)
        print(f"Successfully processed: {basename} (GPU {gpu_id})")
        
        # Rename the infer_cli_out.wav file to the desired name
        temp_output_wav = os.path.join(unique_output_subdir, 'infer_cli_out.wav')
        if os.path.exists(temp_output_wav):
            os.rename(temp_output_wav, output_wav)
            print(f"Renamed: {temp_output_wav} -> {output_wav}")
            # Remove the temporary subdirectory
            os.rmdir(unique_output_subdir)
        else:
            print(f"The output file was not found: {temp_output_wav}")
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing {basename}: {e}")
    
    finally:
        # Remove the converted wav file if it was converted from mp3
        if task['original_ext'] == '.mp3' and task['converted_audio']:
            try:
                os.remove(task['converted_audio'])
                print(f"Deleted the temporary wav file: {task['converted_audio']}")
            except OSError as e:
                print(f"An error occurred while deleting the temporary wav file: {e}")

def main():
    """Main function that initiates parallel processing across multiple GPUs."""
    parser = argparse.ArgumentParser(description="Rewritten f5-tts_infer-cli script with specified parameters.")
    
    # Basic parameters
    parser.add_argument('-c', '--config', help='Configuration file. (optional)')
    parser.add_argument('-m', '--model_type', choices=['F5-TTS', 'E2-TTS'], required=True, help='Model type: F5-TTS or E2-TTS (mandatory)')
    parser.add_argument('-p', '--checkpoint', help='Checkpoint file with .pt extension (optional)')
    parser.add_argument('-v', '--vocab', help='vocab.txt file (optional)')
    
    # Reference audio
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument('-r', '--ref_audio', help='Reference audio file')
    ref_group.add_argument('-rd', '--ref_audio_dir', help='Reference audio directory')
    
    # Reference text
    parser.add_argument('-s', '--ref_text', help='Reference text in "" (mandatory if -rd is not provided)')
    parser.add_argument('-rt', '--ref_text_dir', help='Reference text directory (mandatory if -rd is provided)')
    
    # Generated text
    gen_group = parser.add_mutually_exclusive_group(required=True)
    gen_group.add_argument('-t', '--gen_text', help='Text to generate in ""')
    gen_group.add_argument('-f', '--gen_text_file', help='File containing text to generate')
    gen_group.add_argument('-rt_gen', '--gen_text_dir', help='Reference text directory for generation (mandatory if -rd is provided)')
    
    # Output directory
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for generated files (mandatory)')
    
    # Optional switches
    parser.add_argument('--remove_silence', action='store_true', help='Remove silence (optional)')
    parser.add_argument('--load_vocoder_from_local', action='store_true', help='Load vocoder from local source (optional)')
    parser.add_argument('--speed', type=float, help='Speed (0.1-2) (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_args(args)
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    num_gpus = get_num_gpus()
    print(f"Number of GPUs found: {num_gpus}")
    
    # Gather tasks
    tasks = []
    
    if args.ref_audio:
        # Single reference audio file
        basename = os.path.splitext(os.path.basename(args.ref_audio))[0]
        ref_audio_path = args.ref_audio
        ref_text = args.ref_text
        
        if not os.path.isfile(ref_audio_path):
            print(f"Error: Reference audio file does not exist: {ref_audio_path}")
            sys.exit(1)
        
        if ref_audio_path.lower().endswith('.mp3'):
            # Convert mp3 to wav
            converted_audio = os.path.splitext(ref_audio_path)[0] + '.wav'
            command = ['ffmpeg', '-y', '-i', ref_audio_path, converted_audio]
            try:
                subprocess.run(command, check=True)
                print(f"Converted: {ref_audio_path} -> {converted_audio}")
                ref_audio_final = converted_audio
                converted = True
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while converting mp3: {e}")
                sys.exit(1)
        elif ref_audio_path.lower().endswith('.wav'):
            ref_audio_final = ref_audio_path
            converted = False
        else:
            print(f"Unsupported audio file format: {args.ref_audio}")
            sys.exit(1)
        
        task = {
            'ref_audio': ref_audio_final,
            'ref_text': ref_text,
            'gen_text': args.gen_text,
            'gen_text_file': args.gen_text_file,
            'gen_text_dir': args.gen_text_dir,
            'original_ext': os.path.splitext(args.ref_audio)[1].lower(),
            'converted_audio': converted_audio if args.ref_audio.lower().endswith('.mp3') else None
        }
        tasks.append(task)
    
    elif args.ref_audio_dir:
        # Multiple reference audio files from a directory
        ref_audio_files = [f for f in os.listdir(args.ref_audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
        if not ref_audio_files:
            print(f"Error: No .wav or .mp3 files found in the specified reference directory: {args.ref_audio_dir}")
            sys.exit(1)
        
        for filename in ref_audio_files:
            ref_audio_path = os.path.join(args.ref_audio_dir, filename)
            basename = os.path.splitext(filename)[0]
            
            # Reference text
            if args.ref_text_dir:
                ref_text_file = os.path.join(args.ref_text_dir, f"{basename}.txt")
                if not os.path.isfile(ref_text_file):
                    print(f"Error: Missing text file named {basename}.txt in the reference text directory.")
                    continue
                with open(ref_text_file, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
            else:
                ref_text = args.ref_text
            
            # Conversion if necessary
            if ref_audio_path.lower().endswith('.mp3'):
                converted_audio = os.path.join(args.ref_audio_dir, f"{basename}.wav")
                command = ['ffmpeg', '-y', '-i', ref_audio_path, converted_audio]
                try:
                    subprocess.run(command, check=True)
                    print(f"Converted: {ref_audio_path} -> {converted_audio}")
                    ref_audio_final = converted_audio
                    converted = True
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while converting mp3: {e}")
                    continue
            elif ref_audio_path.lower().endswith('.wav'):
                ref_audio_final = ref_audio_path
                converted = False
            else:
                print(f"Unsupported audio file format: {ref_audio_path}")
                continue
            
            # Generated text
            if args.gen_text_dir:
                gen_text_file = os.path.join(args.gen_text_dir, f"{basename}.txt")
                if not os.path.isfile(gen_text_file):
                    print(f"Error: Missing generated text file named {basename}.txt in the generated text directory.")
                    continue
                with open(gen_text_file, 'r', encoding='utf-8') as f:
                    gen_text = f.read().strip()
                # Assign the generated text directly to the task
                task = {
                    'ref_audio': ref_audio_final,
                    'ref_text': ref_text,
                    'gen_text': gen_text,
                    'gen_text_file': None,
                    'gen_text_dir': args.gen_text_dir,
                    'original_ext': os.path.splitext(filename)[1].lower(),
                    'converted_audio': converted_audio if ref_audio_path.lower().endswith('.mp3') else None
                }
            elif args.gen_text_file:
                # Use the same gen_text_file for all tasks
                task = {
                    'ref_audio': ref_audio_final,
                    'ref_text': ref_text,
                    'gen_text': None,
                    'gen_text_file': args.gen_text_file,
                    'gen_text_dir': args.gen_text_dir,
                    'original_ext': os.path.splitext(filename)[1].lower(),
                    'converted_audio': converted_audio if ref_audio_path.lower().endswith('.mp3') else None
                }
            else:
                # Use direct gen_text
                task = {
                    'ref_audio': ref_audio_final,
                    'ref_text': ref_text,
                    'gen_text': args.gen_text,
                    'gen_text_file': args.gen_text_file,
                    'gen_text_dir': args.gen_text_dir,
                    'original_ext': os.path.splitext(filename)[1].lower(),
                    'converted_audio': converted_audio if ref_audio_path.lower().endswith('.mp3') else None
                }
            
            tasks.append(task)
    
    if not tasks:
        print("No tasks to process.")
        sys.exit(0)
    
    # Assign GPUs to tasks
    tasks_with_gpu = [ (args, task, idx % num_gpus) for idx, task in enumerate(tasks) ]
    
    # Create a pool and distribute the tasks
    with Pool(processes=num_gpus) as pool:
        pool.starmap(process_task_wrapper, tasks_with_gpu)

def process_task_wrapper(args, task, gpu_id):
    """Wrapper for the process_task function to work with Pool.map."""
    process_task(args, task, gpu_id)

if __name__ == "__main__":
    main()

