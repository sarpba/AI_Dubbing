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
    tts_generation,  # Új import
    # Importáld a további tab modulokat itt
)

from tabs.utils import list_projects, get_available_gpus

# A kívánt könyvtárak a gyökérben
work_directory = "workdir"
tts_directory = "TTS"

# Ellenőrzi, hogy a workdir könyvtár létezik-e
if not os.path.exists(work_directory):
    os.makedirs(work_directory)
    print(f'"{work_directory}" könyvtár létrehozva.')
else:
    print(f'"{work_directory}" könyvtár már létezik.')

# Ellenőrzi, hogy a TTS könyvtár létezik-e
if not os.path.exists(tts_directory):
    os.makedirs(tts_directory)
    print(f'"{tts_directory}" könyvtár létrehozva.')
else:
    print(f'"{tts_directory}" könyvtár már létezik.')

def dummy_function(*args, **kwargs):
    return "Ez a funkció még nincs implementálva."

with gr.Blocks() as demo:
    # Definiáljuk a kiválasztott projekt állapotát
    selected_project = gr.State(value=None)

    gr.Markdown("# Automatikus Film Szinkronizáló Alkalmazás")

    # Főablak: Projekt kiválasztása és új projekt létrehozása
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Projekt Kezelés")
            proj_name = gr.Textbox(label="Új Projekt Neve", placeholder="Add meg a projekt nevét")
            video_input = gr.File(label="Film Feltöltése", type="filepath")
            upload_button = gr.Button("Feltöltés és Audio Kivonása")
            output1 = gr.Textbox(label="Eredmény", lines=1)
        with gr.Column(scale=2):
            project_dropdown = gr.Dropdown(label="Kiválasztott Projekt", choices=list_projects(), value=None)

    # Fülek: Minden fül "alablak" lesz, amely a kiválasztott projektet használja
    with gr.Tab("2. Audiosáv beolvasása"):
        gr.Markdown("## Transzkript készítése WhisperX segítségével")

        with gr.Row():
            hf_token = gr.Textbox(label="Hugging Face Token", type="password", placeholder="Add meg a Hugging Face tokened")

        with gr.Row():
            device_selection = gr.Dropdown(label="Eszköz", choices=["cpu", "cuda"], value="cuda")
            device_index_selection = gr.Dropdown(
                label="GPU index",
                choices=[str(i) for i in get_available_gpus()],
                value=str(get_available_gpus()[0]) if get_available_gpus() else "0",
                visible=False
            )

        # Dinamikus láthatóság beállítása
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
            transcribe_button = gr.Button("Transzkripció Indítása")

        output2 = gr.Textbox(label="Eredmény", lines=1)

        transcribe_button.click(
            transcribe_audio_whisperx,
            inputs=[project_dropdown, hf_token, device_selection, device_index_selection],
            outputs=output2
        )

    with gr.Tab("3. Beszéd és Táttérhang Elválasztása"):
        gr.Markdown("## Beszéd és táttérhang elválasztása")

        with gr.Row():
            device_step3 = gr.Radio(label="Eszköz", choices=["cpu", "cuda"], value="cpu")
            keep_full_audio_step3 = gr.Checkbox(label="Teljes audio fájl megtartása", value=False)

        with gr.Row():
            separate_button = gr.Button("Beszéd Leválasztás Indítása")

        output3 = gr.Textbox(label="Eredmény", lines=1)

        separate_button.click(
            separate_audio,
            inputs=[project_dropdown, device_step3, keep_full_audio_step3],
            outputs=output3
        )

    with gr.Tab("4. Audio Darabolás"):
        gr.Markdown("## Audio darabolása JSON és audio fájlok alapján")

        with gr.Row():
            audio_choice = gr.Radio(
                label="Darabolni kívánt audio fájl választása",
                choices=["Teljes audio", "Beszéd eltávolított audio"],
                value="Teljes audio"
            )

        with gr.Row():
            split_button = gr.Button("Audio Darabolása Indítása")

        output4 = gr.Textbox(label="Eredmény", lines=1)

        split_button.click(
            split_audio,
            inputs=[project_dropdown, audio_choice],
            outputs=output4
        )

    with gr.Tab("5. Audio Darabok Ellenőrzése"):
        gr.Markdown("## Audio darabok ellenőrzése WhisperX-szel")

        with gr.Row():
            verify_chunks_button = gr.Button("Beolvasás Indítása")

        output5 = gr.Textbox(label="Eredmény", lines=1)

        verify_chunks_button.click(
            verify_chunks_whisperx,
            inputs=[project_dropdown],
            outputs=output5
        )

    with gr.Tab("5.1. Darabok Összehasonlítása"):
        gr.Markdown("## Darabok összehasonlítása JSON és TXT fájlok alapján")

        with gr.Row():
            compare_button = gr.Button("Összehasonlítás Indítása")

        output51 = gr.HTML(label="Eredmény")

        compare_button.click(
            compare_transcripts_whisperx,
            inputs=[project_dropdown],
            outputs=output51
        )

    with gr.Tab("7. Darabok Fordítása"):
        gr.Markdown("## Chunks gépi fordítása DeepL API segítségével")

        with gr.Row():
            input_language = gr.Dropdown(
                label="Bemeneti Nyelv",
                choices=["EN", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH"],
                value="EN"
            )
            output_language = gr.Dropdown(
                label="Cél Nyelv",
                choices=["EN", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH"],
                value="HU"
            )

        with gr.Row():
            auth_key = gr.Textbox(
                label="DeepL API Kulcs",
                type="password",
                placeholder="Add meg a DeepL API kulcsodat"
            )

        with gr.Row():
            translate_button = gr.Button("Fordítás Indítása")

        output7 = gr.Textbox(label="Eredmény", lines=20)

        translate_button.click(
            translate_chunks,
            inputs=[project_dropdown, input_language, output_language, auth_key],
            outputs=output7
        )

    with gr.Tab("8. TTS Hangfájlok Generálása"):
        gr.Markdown("## TTS hangfájlok generálása a f5-tts_infer-cli parancs segítségével")

        with gr.Row():
            generate_tts_button = gr.Button("TTS Generálás Indítása")

        output8 = gr.Textbox(label="Eredmény", lines=20)

        generate_tts_button.click(
            tts_generation,
            inputs=[project_dropdown],
            outputs=output8
        )

    with gr.Tab("9. Chunks Egyesítése"):
        gr.Markdown("## Darabok egyesítése és audio illesztés, hangerő normalizálás")

        with gr.Row():
            delete_empty_checkbox = gr.Checkbox(label="Üres fájlok törlése (--delete_empty)", value=False)

        with gr.Row():
            db_value_input = gr.Textbox(label="Minimális referencia hangerőszint (dB) (-db)", placeholder="-40.0")
            use_db_checkbox = gr.Checkbox(label="Használja a megadott dB értéket", value=False)

        with gr.Row():
            adjust_audio_button = gr.Button("Audio illesztés és hangerő normalizálás")

        output9_adjust = gr.Textbox(label="Eredmény", lines=20)

        adjust_audio_button.click(
            adjust_audio,
            inputs=[project_dropdown, delete_empty_checkbox, use_db_checkbox, db_value_input],
            outputs=output9_adjust
        )

        with gr.Row():
            merge_chunks_button = gr.Button("Chunks Egyesítése Indítása")

        output9 = gr.Textbox(label="Eredmény", lines=20)

        merge_chunks_button.click(
            merge_chunks,
            inputs=[project_dropdown],
            outputs=output9
        )

    with gr.Tab("10. Audio Integrálása a Videóba"):
        gr.Markdown("## Audio fájl integrálása a videóba a megadott nyelvi címkével")

        with gr.Row():
            language_code = gr.Textbox(label="Nyelvi kód", placeholder="pl. eng, hun", value="hun")

        with gr.Row():
            integrate_audio_button = gr.Button("Audio Integrálása Indítása")

        output10 = gr.Textbox(label="Eredmény", lines=20)

        integrate_audio_button.click(
            integrate_audio,
            inputs=[project_dropdown, language_code],
            outputs=output10
        )

    gr.Markdown("""
    ---
    **Megjegyzés:** Ez a projekt, egy kisérleti MI szinkron készítő. Fejlesztés alatt, de lehet, hogy nem.
    """)

    # Függvény a projekt létrehozására és Dropdown frissítésére
    def create_project_and_update(proj_name, video_input):
        if not proj_name:
            return "Hiba: Projekt név nem lehet üres.", gr.update(choices=list_projects(), value=selected_project.value)
        # Létrehozzuk a projektet és kinyerjük az audio fájlt
        result = upload_and_extract_audio(proj_name, video_input)
        
        # Frissítjük a projekt listát
        projects = list_projects()
        
        # Frissítjük a kiválasztott projektot az új projektre
        selected_project.value = proj_name
        
        # Frissítjük a Dropdown választásait és kiválasztjuk az új projektet
        dropdown_update = gr.update(choices=projects, value=proj_name)
        
        return result, dropdown_update

    # Fülön kívüli, főablakban található feltöltés és projekt létrehozás
    upload_button.click(
        create_project_and_update,
        inputs=[proj_name, video_input],
        outputs=[output1, project_dropdown]
    )

    # Alkalmazás betöltésekor inicializáljuk a projekt kiválasztást
    def initialize_projects():
        projects = list_projects()
        if not projects:
            selected_project.value = None
            return gr.update(choices=[], value=None)
        selected_project.value = projects[0]
        return gr.update(choices=projects, value=projects[0])

    demo.load(fn=initialize_projects, inputs=[], outputs=project_dropdown)

# Alkalmazás indítása
demo.launch()
