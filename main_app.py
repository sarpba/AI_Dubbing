# main_app.py
import os
import gradio as gr
import argparse
import socket
from tabs import (
    upload_and_extract_audio,
    transcribe_audio_whisperx,
    separate_audio,
    split_audio,
    verify_chunks_whisperx,
    compare_transcripts_whisperx,
    translate_chunks,
    merge_chunks,
    integrate_audio,
    adjust_audio,
    tts_generation,  # New import
    # Import additional tab modules here
)

from tabs.utils import list_projects, get_available_gpus, get_available_demucs_models

# Desired directories at the root
work_directory = "workdir"
tts_directory = "TTS"

# Check if the workdir directory exists
if not os.path.exists(work_directory):
    os.makedirs(work_directory)
    print(f'"{work_directory}" directory created.')
else:
    print(f'"{work_directory}" directory already exists.')

# Check if the TTS directory exists
if not os.path.exists(tts_directory):
    os.makedirs(tts_directory)
    print(f'"{tts_directory}" directory created.')
else:
    print(f'"{tts_directory}" directory already exists.')

def dummy_function(*args, **kwargs):
    return "This function is not yet implemented."

with gr.Blocks() as demo:
    # Define the selected project state
    selected_project = gr.State(value=None)

    gr.Markdown("# Automatic Movie Dubbing Application")

    # Main window: Project selection and new project creation
    with gr.Row():

        with gr.Column(scale=1):
            project_dropdown = gr.Dropdown(label="Selected Project", choices=list_projects(), value=None)

    # Tabs: Each tab is a sub-window that uses the selected project
    with gr.Tab("1. New Project"):
        gr.Markdown("## Create a new Project")

        with gr.Row():
            with gr.Column(scale=1):
                proj_name = gr.Textbox(label="New Project Name", placeholder="Enter the project name")
                video_input = gr.File(label="Upload Movie", type="filepath")
                upload_button = gr.Button("Upload and Extract Audio")
                output1 = gr.Textbox(label="Result", lines=1)         


    with gr.Tab("2. Separate Speech"):
        gr.Markdown("## Separate Speech and Background Sound")

        with gr.Row():
            device_step3 = gr.Radio(label="Device", choices=["cpu", "cuda"], value="cuda")
            keep_full_audio_step3 = gr.Checkbox(label="Keep Full Audio File", value=False)

        # Add hozzá a Demucs modell választó legördülő menüt
        with gr.Row():
            demucs_model_selection = gr.Dropdown(
                label="Select Demucs Model",
                choices=get_available_demucs_models(),
                value="htdemucs"
            )

        with gr.Row():
            separate_button = gr.Button("Start Speech Separation")

        output3 = gr.Textbox(label="Result", lines=1)

        # Frissítsd a separate_button.click hívást az új bemenettel
        separate_button.click(
            separate_audio,
            inputs=[project_dropdown, device_step3, keep_full_audio_step3, demucs_model_selection],
            outputs=output3
        )


    with gr.Tab("3. Read In"):
        gr.Markdown("## Create Transcript with WhisperX")

        with gr.Row():
            hf_token = gr.Textbox(
                label="Hugging Face Token (If you want to use speaker diarization)",
                type="password",
                placeholder="Enter your Hugging Face token"
            )
            language = gr.Textbox(
                label="FORCE Whisper Language (if you not set, it'll try to autodetect)",
                placeholder="Enter language (e.g., 'en', 'hu')"
            )

        with gr.Row():
            device_selection = gr.Dropdown(
                label="Device",
                choices=["cpu", "cuda"],
                value="cuda"
            )
            device_index_selection = gr.Dropdown(
                label="GPU Index",
                choices=[str(i) for i in get_available_gpus()],
                value=str(get_available_gpus()[0]) if get_available_gpus() else "0",
                visible=False
            )

        # Dynamic visibility setting for GPU index
        def update_device_index_visibility(device):
            if device == "cuda":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        device_selection.change(
            update_device_index_visibility,
            inputs=device_selection,
            outputs=device_index_selection
        )

        with gr.Row():
            # Új Dropdown a mappa kiválasztásához
            audio_source_selection = gr.Dropdown(
                label="Select Audio Source",
                choices=["audio", "speech_removed"],
                value="speech_removed"
            )

        with gr.Row():
            transcribe_button = gr.Button("Start Transcription")

        output2 = gr.Textbox(label="Result", lines=10)

        transcribe_button.click(
            transcribe_audio_whisperx,
            inputs=[
                project_dropdown,
                hf_token,
                device_selection,
                device_index_selection,
                audio_source_selection,
                language
            ],
            outputs=output2
        )


    with gr.Tab("3.1. Json reworking"):
        gr.Markdown("## Splitting long sentences. (not implemented jet)")


    with gr.Tab("4. Audio Splitting"):
        gr.Markdown("## Split Audio Based on JSON and Audio Files")

        with gr.Row():
            audio_choice = gr.Radio(
                label="Select Audio File to Split",
                choices=["Full Audio", "Speech Only"],
                value="Speech Only"
            )

        with gr.Row():
            split_button = gr.Button("Start Audio Splitting")

        output4 = gr.Textbox(label="Result", lines=1)

        split_button.click(
            split_audio,
            inputs=[project_dropdown, audio_choice],
            outputs=output4
        )

    with gr.Tab("5. Read Chunks"):
        gr.Markdown("## Read Audio Chunks with WhisperX")

        with gr.Row():
            verify_chunks_button = gr.Button("Start Verification")

        output5 = gr.Textbox(label="Result", lines=1)

        verify_chunks_button.click(
            verify_chunks_whisperx,
            inputs=[project_dropdown],
            outputs=output5
        )

    with gr.Tab("5.1. Compare Chunks"):
        gr.Markdown("## Compare Chunks Based on TXT from first and JSON from second Read process")

        with gr.Row():
            compare_button = gr.Button("Start Comparison")

        # States to hold the comparison data and current page
        comparison_data_state = gr.State([])
        current_page_state = gr.State(0)

        # Number of items per page
        items_per_page = 5  # Adjust as needed

        # Error message output
        error_output = gr.Markdown(value="", visible=False)
        # Save message output
        save_message_output = gr.Markdown(value="", visible=False)

        # Define components for each item
        components = []
        for i in range(items_per_page):
            with gr.Row(visible=False) as row:
                # Timestamp (First Column)
                timestamp = gr.Textbox(label="Timestamp", interactive=False, max_lines=1, scale=1)
                # JSON Text (Second Column)
                json_text = gr.Textbox(label="JSON Text", interactive=False, scale=2)
                # TXT Text
                txt_text = gr.Textbox(label="TXT Text", interactive=False, scale=2)
                # Audio Player 1
                audio_player = gr.Audio(label="Audio", scale=1)
                # Translated TXT (editable)
                translated_txt = gr.Textbox(label="Translated TXT", interactive=True, scale=2)
                # Audio Player 2
                sync_audio_player = gr.Audio(label="Sync Audio", scale=1)
                # Save Button
                save_button = gr.Button("Save", scale=1)
                components.append({
                    'row': row,
                    'timestamp': timestamp,
                    'json_text': json_text,
                    'txt_text': txt_text,
                    'audio_player': audio_player,
                    'translated_txt': translated_txt,
                    'sync_audio_file': sync_audio_player,
                    'save_button': save_button,
                    'index': i
                })

        # Navigation buttons
        with gr.Row():
            prev_button = gr.Button("Previous")
            next_button = gr.Button("Next")

        # Function to update the display based on current page
        def update_display(data_list, page):
            start = page * items_per_page
            end = start + items_per_page
            page_items = data_list[start:end]

            updates = []
            for i, component in enumerate(components):
                if i < len(page_items):
                    data_item = page_items[i]
                    # Update visibility
                    updates.append(gr.update(visible=True))  # Row
                    # Update components
                    updates.append(gr.update(value=data_item['timestamp']))  # timestamp
                    updates.append(gr.update(value=data_item['json_text']))  # json_text
                    updates.append(gr.update(value=data_item['txt_text']))  # txt_text
                    updates.append(gr.update(value=data_item['audio_file']))  # audio_player
                    updates.append(gr.update(value=data_item['translated_txt']))  # translated_txt
                    updates.append(gr.update(value=data_item['sync_audio_file']))  # sync_audio_player
                    updates.append(gr.update(visible=True))  # save_button
                else:
                    # Hide unused components
                    updates.append(gr.update(visible=False))  # Row
                    updates.append(gr.update(value=""))  # timestamp
                    updates.append(gr.update(value=""))  # json_text
                    updates.append(gr.update(value=""))  # txt_text
                    updates.append(gr.update(value=None))  # audio_player
                    updates.append(gr.update(value=""))  # translated_txt
                    updates.append(gr.update(value=None))  # sync_audio_player
                    updates.append(gr.update(visible=False))  # save_button
            return updates

        # Flatten the list of outputs
        output_components = []
        for component in components:
            output_components.extend([
                component['row'],
                component['timestamp'],
                component['json_text'],
                component['txt_text'],
                component['audio_player'],
                component['translated_txt'],
                component['sync_audio_file'],
                component['save_button']
            ])

        # Compare button action
        def on_compare_button_click(proj_name):
            data_list, error_message = compare_transcripts_whisperx(proj_name)
            if error_message:
                # Create empty updates for all components
                empty_updates = [gr.update(visible=False) for _ in output_components]
                return [gr.update(value=error_message, visible=True), data_list, 0] + empty_updates
            else:
                updates = update_display(data_list, 0)
                return [gr.update(value="", visible=False), data_list, 0] + updates

        compare_button.click(
            fn=on_compare_button_click,
            inputs=[project_dropdown],
            outputs=[error_output, comparison_data_state, current_page_state] + output_components
        )

        # Next page action
        def on_next_button_click(current_page, data_list):
            max_page = (len(data_list) - 1) // items_per_page
            new_page = min(current_page + 1, max_page)
            updates = update_display(data_list, new_page)
            return [new_page] + updates

        next_button.click(
            fn=on_next_button_click,
            inputs=[current_page_state, comparison_data_state],
            outputs=[current_page_state] + output_components
        )

        # Previous page action
        def on_prev_button_click(current_page, data_list):
            new_page = max(current_page - 1, 0)
            updates = update_display(data_list, new_page)
            return [new_page] + updates

        prev_button.click(
            fn=on_prev_button_click,
            inputs=[current_page_state, comparison_data_state],
            outputs=[current_page_state] + output_components
        )

        # Save button action
        def save_translated_text(proj_name, data_list, page, translated_text, item_index):
            # Calculate the actual index in data_list
            actual_index = page * items_per_page + item_index
            if actual_index < len(data_list):
                data_item = data_list[actual_index]
                # Update the translated text in data_item
                data_item['translated_txt'] = translated_text
                # Save the text back to the file
                translated_txt_path = data_item['translated_txt_path']
                try:
                    with open(translated_txt_path, 'w', encoding='utf-8') as f:
                        f.write(translated_text)
                    return gr.update(value="Changes saved successfully.", visible=True)
                except Exception as e:
                    return gr.update(value=f"Error saving changes: {str(e)}", visible=True)
            else:
                return gr.update(value="Invalid item index.", visible=True)

        # Save button click events
        for idx, component in enumerate(components):
            item_save_button = component['save_button']
            item_translated_txt = component['translated_txt']
            item_index = idx  # Index within the page (0 to items_per_page - 1)

            # Save button click event
            item_save_button.click(
                fn=save_translated_text,
                inputs=[
                    project_dropdown,
                    comparison_data_state,
                    current_page_state,
                    item_translated_txt,
                    gr.State(item_index)
                ],
                outputs=save_message_output
            )

        # Display the save message output
        with gr.Row():
            save_message_output


    with gr.Tab("7. Translate Chunks"):
        gr.Markdown("## Machine Translation of Chunks Using DeepL API")

        with gr.Row():
            input_language = gr.Dropdown(
                label="Input Language",
                choices=["EN", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH", "JA"],
                value="EN"
            )
            output_language = gr.Dropdown(
                label="Target Language",
                choices=["EN-US", "EN-UK", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH"],
                value="HU"
            )

        with gr.Row():
            auth_key = gr.Textbox(
                label="DeepL API Key",
                type="password",
                placeholder="Enter your DeepL API key"
            )

        with gr.Row():
            translate_button = gr.Button("Start Translation")

        output7 = gr.Textbox(label="Result", lines=1)

        translate_button.click(
            translate_chunks,
            inputs=[project_dropdown, input_language, output_language, auth_key],
            outputs=output7
        )

    with gr.Tab("8. Generate TTS Audio Files"):
        gr.Markdown("## Generate TTS Audio Files Using f5-tts_infer-cli Command")

        # A TTS mappa alkönyvtárainak beolvasása
        tts_directory = "TTS"
        subdirectories = [d for d in os.listdir(tts_directory) if os.path.isdir(os.path.join(tts_directory, d))]

        with gr.Row():
            tts_language = gr.Dropdown(
                label="Target Language (put the.pt and vocab.txt file the /TTS/your_folder)",
                choices=subdirectories,
                value=None  # Nincs alapértelmezett érték
            )

        with gr.Row():
            generate_tts_button = gr.Button("Start TTS Generation")

        output8 = gr.Textbox(label="Result", lines=1)

        generate_tts_button.click(
            tts_generation,
            inputs=[project_dropdown, tts_language],
            outputs=output8
        )

    with gr.Tab("9. Merge Chunks"):
        gr.Markdown("## Merge Chunks and Adjust Audio with Volume Normalization")

        with gr.Row():
            delete_empty_checkbox = gr.Checkbox(label="Delete Empty Files (--delete_empty)", value=False)

        with gr.Row():
            db_value_input = gr.Textbox(label="Minimum Reference Volume Level (dB) (-db)", placeholder="-40.0")
            use_db_checkbox = gr.Checkbox(label="Use Specified dB Value", value=False)
        
        gr.Markdown("## For more accurate timestamps, need to use ( 5. Read Chunks) tab again before continue")
        with gr.Row():
            adjust_audio_button = gr.Button("Adjust Audio and Normalize Volume")

        output9_adjust = gr.Textbox(label="Result", lines=1)

        adjust_audio_button.click(
            adjust_audio,
            inputs=[project_dropdown, delete_empty_checkbox, use_db_checkbox, db_value_input],
            outputs=output9_adjust
        )

        with gr.Row():
            merge_chunks_button = gr.Button("Start Merging Chunks")

        output9 = gr.Textbox(label="Result", lines=1)

        merge_chunks_button.click(
            merge_chunks,
            inputs=[project_dropdown],
            outputs=output9
        )

    with gr.Tab("10. Integrate Audio into Video"):
        gr.Markdown("## Integrate Audio File into Video with Specified Language Tag")

        with gr.Row():
            language_code = gr.Textbox(label="Language Code", placeholder="e.g., eng, hun", value="hun")

        with gr.Row():
            integrate_audio_button = gr.Button("Start Audio Integration")

        output10 = gr.Textbox(label="Result", lines=1)

        integrate_audio_button.click(
            integrate_audio,
            inputs=[project_dropdown, language_code],
            outputs=output10
        )

    gr.Markdown("""
    ---
    **Note:** This project is an experimental AI dubbing creator. Under development, but maybe not continue, don't know.
    """)

    # Function to create a project and update the Dropdown
    def create_project_and_update(proj_name, video_input):
        if not proj_name:
            return "Error: Project name cannot be empty.", gr.update(choices=list_projects(), value=selected_project.value)
        # Create the project and extract the audio file
        result = upload_and_extract_audio(proj_name, video_input)
        
        # Update the project list
        projects = list_projects()
        
        # Update the selected project to the new project
        selected_project.value = proj_name
        
        # Update the Dropdown choices and select the new project
        dropdown_update = gr.update(choices=projects, value=proj_name)
        
        return result, dropdown_update

    # Upload and create project in the main window, outside tabs
    upload_button.click(
        create_project_and_update,
        inputs=[proj_name, video_input],
        outputs=[output1, project_dropdown]
    )

    # Initialize project selection when the app loads
    def initialize_projects():
        projects = list_projects()
        if not projects:
            selected_project.value = None
            return gr.update(choices=[], value=None)
        selected_project.value = projects[0]
        return gr.update(choices=projects, value=projects[0])

    demo.load(fn=initialize_projects, inputs=[], outputs=project_dropdown)

## Argument parsers setup
def parse_args():
    parser = argparse.ArgumentParser(description="Run Gradio app with optional share, port, and host settings.")
    parser.add_argument('--share', action='store_true', help='Create a shareable URL for Gradio.')
    parser.add_argument('--port', type=int, default=7860, help='Port number for the application to run on (default: 7860).')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address for the application to bind to (default: 127.0.0.1). If you want to share on local lan just use 0.0.0.0')
    return parser.parse_args()

# Port checker function
def is_port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            s.close()
            return True
        except socket.error:
            return False

# Main section
if __name__ == "__main__":
    args = parse_args()

    port = args.port
    host = args.host
    initial_port = port

    max_attempts = 10  # Maximum number of port attempts
    attempt = 0

    while not is_port_free(host, port) and attempt < max_attempts:
        print(f"Error: Port {port} is occupied on host {host}. Attempting to start on port {port + 1}.")
        port += 1
        attempt += 1

    if not is_port_free(host, port):
        print(f"Error: Could not find a free port in the range {initial_port} to {port + max_attempts}. The application cannot start.")
        exit(1)
    else:
        if port != initial_port:
            print(f"Gradio application is starting on host {host} and port {port}.")
        else:
            print(f"Gradio application is starting on host {host} and port {port}.")

    # Launch the application with share, port, and host options
    demo.launch(share=args.share, server_port=port, server_name=host)