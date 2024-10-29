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


# A kívánt könyvtár munkakönyvtár
directory = "workdir"

# Ellenőrzi, hogy a könyvtár létezik-e
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f'"{directory}" könyvtár létrehozva.')
else:
    print(f'"{directory}" könyvtár már létezik.')


from tabs.utils import list_projects, get_available_gpus

def dummy_function(*args, **kwargs):
    return "Ez a funkció még nincs implementálva."

with gr.Blocks() as demo:
    gr.Markdown("# Automatikus Film Szinkronizáló Alkalmazás")

    # Tab 1: Projekt létrehozása és film feltöltése
    with gr.Tab("1. Projekt Létrehozása és Film Feltöltése"):
        gr.Markdown("## Projekt létrehozása és film feltöltése")

        with gr.Row():
            proj_name = gr.Textbox(label="Projekt neve", placeholder="Add meg a projekt nevét")

        with gr.Row():
            video_input = gr.File(label="Film feltöltése", type="filepath")  # max_size eltávolítva

        with gr.Row():
            upload_button = gr.Button("Feltöltés és Audio kivonása")

        output1 = gr.Textbox(label="Eredmény", lines=4)

        upload_button.click(
            upload_and_extract_audio,
            inputs=[proj_name, video_input],
            outputs=output1
        )

    # Tab 2: Transzkript készítés (WhisperX) külső script hívásával
    with gr.Tab("2. Audiosáv beolvasása"):
        gr.Markdown("## Transzkript készítése WhisperX segítségével")

        with gr.Row():
            proj_name_step2 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

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

        output2 = gr.Textbox(label="Eredmény", lines=20)

        transcribe_button.click(
            transcribe_audio_whisperx,
            inputs=[proj_name_step2, hf_token, device_selection, device_index_selection],
            outputs=output2
        )

    # Tab 3: Beszéd Eltávolítása
    with gr.Tab("3. Beszéd és táttérhang elválasztása"):
        gr.Markdown("## Beszéd és táttérhang elválasztása")

        with gr.Row():
            proj_name_step3 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

        with gr.Row():
            device_step3 = gr.Radio(label="Eszköz", choices=["cpu", "cuda"], value="cpu")
            keep_full_audio_step3 = gr.Checkbox(label="Teljes audio fájl megtartása", value=False)

        with gr.Row():
            separate_button = gr.Button("Beszéd Leválasztás Indítása")

        output3 = gr.Textbox(label="Eredmény", lines=20)

        separate_button.click(
            separate_audio,
            inputs=[proj_name_step3, device_step3, keep_full_audio_step3],
            outputs=output3
        )

    # Tab 4: Audio Darabolás
    with gr.Tab("4. Audio Darabolás"):
        gr.Markdown("## Audio darabolása JSON és audio fájlok alapján")

        with gr.Row():
            proj_name_step4 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

        with gr.Row():
            audio_choice = gr.Radio(
                label="Darabolni kívánt audio fájl választása",
                choices=["Teljes audio", "Beszéd eltávolított audio"],
                value="Teljes audio"
            )

        with gr.Row():
            split_button = gr.Button("Audio Darabolása Indítása")

        output4 = gr.Textbox(label="Eredmény", lines=20)

        split_button.click(
            split_audio,
            inputs=[proj_name_step4, audio_choice],
            outputs=output4
        )

    # Tab 5: Chunks Ellenőrzése (WhisperX-sel)
    with gr.Tab("5. Audio darabok ellenőrzése"):
        gr.Markdown("## Audio darabok ellenőrzése WhisperX-szel")

        with gr.Row():
            proj_name_step5 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

        with gr.Row():
            verify_chunks_button = gr.Button("Beolvasás Indítása")

        output5 = gr.Textbox(label="Eredmény", lines=20)

        verify_chunks_button.click(
            verify_chunks_whisperx,
            inputs=[proj_name_step5],
            outputs=output5
        )

    # Tab 5.1: Chunks Összehasonlítása
    with gr.Tab("5.1. Darabok Összehasonlítása"):
        gr.Markdown("## Darabok összehasonlítása JSON és TXT fájlok alapján")

        with gr.Row():
            proj_name_step51 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

        with gr.Row():
            compare_button = gr.Button("Összehasonlítás Indítása")

        output51 = gr.HTML(label="Eredmény")

        compare_button.click(
            compare_transcripts_whisperx,
            inputs=[proj_name_step51],
            outputs=output51
        )

    # Tab 7: Chunks Fordítása
    with gr.Tab("7. Darabok Fordítása"):
        gr.Markdown("## Chunks gépi fordítása DeepL API segítségével")

        with gr.Row():
            proj_name_step7 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

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
            inputs=[proj_name_step7, input_language, output_language, auth_key],
            outputs=output7
        )

    # Tab 8: TTS Hangfájlok Generálása
    with gr.Tab("8. TTS Hangfájlok Generálása"):
        gr.Markdown("## TTS hangfájlok generálása a F5-TTS segítségével")

        with gr.Row():
            proj_name_step8 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

        with gr.Row():
            generate_tts_button = gr.Button("TTS Generálás Indítása")

        output8 = gr.Textbox(label="Eredmény", lines=20)

        generate_tts_button.click(
            tts_generation,
            inputs=[proj_name_step8],
            outputs=output8
        )

    # Tab 9: Chunks Egyesítése
    with gr.Tab("9. Chunks Egyesítése"):
        gr.Markdown("## Darabok egyesítése és audio illesztés, hangerő normalizálás")

        with gr.Row():
            proj_name_step9 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

        gr.Markdown("## Audio illesztés és hangerő normalizálás")

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
            inputs=[proj_name_step9, delete_empty_checkbox, use_db_checkbox, db_value_input],
            outputs=output9_adjust
        )

        with gr.Row():
            merge_chunks_button = gr.Button("Chunks Egyesítése Indítása")

        output9 = gr.Textbox(label="Eredmény", lines=20)

        merge_chunks_button.click(
            merge_chunks,
            inputs=[proj_name_step9],
            outputs=output9
        )
    # Tab 10: Audio Integrálása a Videóba
    with gr.Tab("10. Audio Integrálása a Videóba"):
        gr.Markdown("## Audio fájl integrálása a videóba a megadott nyelvi címkével")

        with gr.Row():
            proj_name_step10 = gr.Dropdown(label="Projekt kiválasztása", choices=list_projects(), interactive=True)

        with gr.Row():
            language_code = gr.Textbox(label="Nyelvi kód", placeholder="pl. eng, hun", value="hun")

        with gr.Row():
            integrate_audio_button = gr.Button("Audio Integrálása Indítása")

        output10 = gr.Textbox(label="Eredmény", lines=20)

        integrate_audio_button.click(
            integrate_audio,
            inputs=[proj_name_step10, language_code],
            outputs=output10
        )


    gr.Markdown("""
    ---
    **Megjegyzés:** Ez a projekt, egy kisérleti MI sznkron készítő. Fejlesztés alatt, de lehet, hogy nem.
    """)

    # Assign references to Dropdowns
    project_dropdowns = [
        proj_name_step2,
        proj_name_step3,
        proj_name_step4,
        proj_name_step5,
        proj_name_step51,
        proj_name_step7,
        proj_name_step8,
        proj_name_step9,
        proj_name_step10
    ]

    # Frissítjük a projekt kiválasztási dropdown listáit, amikor az alkalmazás betöltődik
    def update_project_dropdowns(workdir="workdir"):
        projects = list_projects(workdir)
        return [gr.update(choices=projects) for _ in project_dropdowns]

    # Az alkalmazás betöltésekor frissítjük a Dropdownokat
    demo.load(fn=update_project_dropdowns, inputs=[], outputs=project_dropdowns)

# Alkalmazás indítása
demo.launch()

