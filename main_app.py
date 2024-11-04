# main_app.py
import os
import gradio as gr
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

from tabs.utils import list_projects, get_available_gpus

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
            gr.Markdown("## Project Management")
            proj_name = gr.Textbox(label="New Project Name", placeholder="Enter the project name")
            video_input = gr.File(label="Upload Movie", type="filepath")
            upload_button = gr.Button("Upload and Extract Audio")
            output1 = gr.Textbox(label="Result", lines=1)
        with gr.Column(scale=2):
            project_dropdown = gr.Dropdown(label="Selected Project", choices=list_projects(), value=None)

    # Tabs: Each tab is a sub-window that uses the selected project
    with gr.Tab("2. Read Audio Track"):
        gr.Markdown("## Create Transcript with WhisperX")

        with gr.Row():
            hf_token = gr.Textbox(label="Hugging Face Token", type="password", placeholder="Enter your Hugging Face token")
            language = gr.Textbox(label="Language", placeholder="Enter language (e.g., 'en', 'hu')")

        with gr.Row():
            device_selection = gr.Dropdown(label="Device", choices=["cpu", "cuda"], value="cuda")
            device_index_selection = gr.Dropdown(
                label="GPU Index",
                choices=[str(i) for i in get_available_gpus()],
                value=str(get_available_gpus()[0]) if get_available_gpus() else "0",
                visible=False
            )

        # Dynamic visibility setting
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
            transcribe_button = gr.Button("Start Transcription")

        output2 = gr.Textbox(label="Result", lines=1)

        transcribe_button.click(
            transcribe_audio_whisperx,
            inputs=[project_dropdown, hf_token, device_selection, device_index_selection, language],
            outputs=output2
        )

    with gr.Tab("3. Separate Speech and Background Sound"):
        gr.Markdown("## Separate Speech and Background Sound")

        with gr.Row():
            device_step3 = gr.Radio(label="Device", choices=["cpu", "cuda"], value="cpu")
            keep_full_audio_step3 = gr.Checkbox(label="Keep Full Audio File", value=False)

        with gr.Row():
            separate_button = gr.Button("Start Speech Separation")

        output3 = gr.Textbox(label="Result", lines=1)

        separate_button.click(
            separate_audio,
            inputs=[project_dropdown, device_step3, keep_full_audio_step3],
            outputs=output3
        )

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

    with gr.Tab("5. Verify Audio Chunks"):
        gr.Markdown("## Verify Audio Chunks with WhisperX")

        with gr.Row():
            verify_chunks_button = gr.Button("Start Verification")

        output5 = gr.Textbox(label="Result", lines=1)

        verify_chunks_button.click(
            verify_chunks_whisperx,
            inputs=[project_dropdown],
            outputs=output5
        )

    with gr.Tab("5.1. Compare Chunks"):
        gr.Markdown("## Compare Chunks Based on JSON and TXT Files")

        with gr.Row():
            compare_button = gr.Button("Start Comparison")

        output51 = gr.HTML(label="Result")

        compare_button.click(
            compare_transcripts_whisperx,
            inputs=[project_dropdown],
            outputs=output51
        )

    with gr.Tab("7. Translate Chunks"):
        gr.Markdown("## Machine Translation of Chunks Using DeepL API")

        with gr.Row():
            input_language = gr.Dropdown(
                label="Input Language",
                choices=["EN", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH"],
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

        with gr.Row():
            generate_tts_button = gr.Button("Start TTS Generation")

        output8 = gr.Textbox(label="Result", lines=1)

        generate_tts_button.click(
            tts_generation,
            inputs=[project_dropdown],
            outputs=output8
        )

    with gr.Tab("9. Merge Chunks"):
        gr.Markdown("## Merge Chunks and Adjust Audio with Volume Normalization")

        with gr.Row():
            delete_empty_checkbox = gr.Checkbox(label="Delete Empty Files (--delete_empty)", value=False)

        with gr.Row():
            db_value_input = gr.Textbox(label="Minimum Reference Volume Level (dB) (-db)", placeholder="-40.0")
            use_db_checkbox = gr.Checkbox(label="Use Specified dB Value", value=False)

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
    **Note:** This project is an experimental AI dubbing creator. Under development, but maybe not.
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

# Launch the application
demo.launch()