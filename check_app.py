#!/usr/bin/env python3
import os
import sys
import json
import math
import logging
import gradio as gr
from mutagen import File as MutagenFile

############################################
# --project kapcsoló kezelése
############################################

default_project = None
if "--project" in sys.argv:
    idx = sys.argv.index("--project")
    if idx + 1 < len(sys.argv):
        default_project = sys.argv[idx + 1]

############################################
# config.json beolvasása
############################################

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

############################################
# Logger globális, de a fileHandler-t
# mindig az éppen kiválasztott projekthez igazítjuk.
############################################

logger = logging.getLogger("CheckerApp")
logger.setLevel(logging.INFO)

def setup_project_logger(project: str):
    """
    A megadott projekt logs könyvtárába állítjuk be a fileHandler-t,
    config["PROJECT_SUBDIRS"]["logs"] = "logs" alapján.
    """
    # Eltávolítjuk a régi FileHandler-t, ha van:
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    workdir = config["DIRECTORIES"]["workdir"]
    logs_subdir = config["PROJECT_SUBDIRS"]["logs"]
    project_logs_dir = os.path.join(workdir, project, logs_subdir)
    os.makedirs(project_logs_dir, exist_ok=True)

    log_file = os.path.join(project_logs_dir, "checker_app.log")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=== Logger set up for project: %s ===", project)

############################################
# Segédfüggvények
############################################

ROW_COUNT = 10  # 10 sor / lap

def parse_timestamp(ts_str):
    parts = ts_str.split('-')
    if len(parts) < 3:
        return 0.0
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except Exception as e:
        logger.warning(f"parse_timestamp hiba: {ts_str}, {e}")
        return 0.0

def load_project_data(project):
    """
    Betölti a splits és translated_splits mappákból a .wav/.txt fájlokat,
    elkészíti a "row" listát (overlong check).
    """
    logger.info(f"Loading data for project='{project}'")
    workdir = config["DIRECTORIES"]["workdir"]
    project_dir = os.path.join(workdir, project)
    splits_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["splits"])
    trans_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["translated_splits"])

    data = []
    if not os.path.isdir(splits_dir):
        logger.warning(f"Splits dir not found: {splits_dir}")
        return data

    for file in os.listdir(splits_dir):
        if file.endswith(".wav"):
            basename = file[:-4]
            splits_wav = os.path.join(splits_dir, basename + ".wav")
            splits_txt_path = os.path.join(splits_dir, basename + ".txt")
            splits_txt = ""
            if os.path.exists(splits_txt_path):
                with open(splits_txt_path, "r", encoding="utf-8") as tf:
                    splits_txt = tf.read()

            trans_wav_path = os.path.join(trans_dir, basename + ".wav")
            trans_wav = trans_wav_path if os.path.exists(trans_wav_path) else None
            trans_txt_path = os.path.join(trans_dir, basename + ".txt")
            trans_txt = ""
            if os.path.exists(trans_txt_path):
                with open(trans_txt_path, "r", encoding="utf-8") as tf:
                    trans_txt = tf.read()

            # Timestamp
            ts_part = basename.split('_')[0]
            start_time = parse_timestamp(ts_part)

            data.append({
                "basename": basename,
                "splits_wav": splits_wav,
                "splits_txt": splits_txt,
                "trans_wav": trans_wav,
                "trans_txt": trans_txt,
                "start": start_time
            })

    data.sort(key=lambda x: x["start"])
    for i in range(len(data) - 1):
        data[i]["allowed_interval"] = data[i+1]["start"] - data[i]["start"]
    if data:
        data[-1]["allowed_interval"] = None

    # Overlong check
    for row in data:
        row["overlong"] = False
        row["diff"] = 0.0
        if row["trans_wav"] and row.get("allowed_interval") is not None:
            try:
                audio = MutagenFile(row["trans_wav"])
                duration = audio.info.length
            except:
                duration = 0.0
            row["trans_duration"] = duration
            if duration > row["allowed_interval"]:
                row["overlong"] = True
                row["diff"] = duration - row["allowed_interval"]
        else:
            row["trans_duration"] = 0.0

    logger.info(f"Loaded {len(data)} splitted items in project={project}.")
    return data

def delete_audio_and_refresh(project, basename, page):
    """
    Törli a translated .wav + .json fájlokat, reload.
    """
    logger.info(f"Deleting translated audio & JSON: project={project}, basename={basename}")
    workdir = config["DIRECTORIES"]["workdir"]
    trans_dir = os.path.join(workdir, project, config["PROJECT_SUBDIRS"]["translated_splits"])

    wav_path = os.path.join(trans_dir, basename + ".wav")
    json_path = os.path.join(trans_dir, basename + ".json")

    if os.path.exists(wav_path):
        os.remove(wav_path)
        logger.info(f"Deleted file: {wav_path}")
    if os.path.exists(json_path):
        os.remove(json_path)
        logger.info(f"Deleted JSON file: {json_path}")

    data = load_project_data(project)
    return data, page

def save_text_and_refresh(project, basename, new_text, page):
    """
    Elmenti az új trans_txt, törli a .wav + .json, reload.
    """
    logger.info(f"Saving text => project={project}, basename={basename}, text_len={len(new_text)}")
    workdir = config["DIRECTORIES"]["workdir"]
    trans_dir = os.path.join(workdir, project, config["PROJECT_SUBDIRS"]["translated_splits"])

    txt_path = os.path.join(trans_dir, basename + ".txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(new_text)
    logger.info(f"Updated .txt => {txt_path}")

    wav_path = os.path.join(trans_dir, basename + ".wav")
    if os.path.exists(wav_path):
        os.remove(wav_path)
        logger.info(f"Removed wav => {wav_path}")

    json_path = os.path.join(trans_dir, basename + ".json")
    if os.path.exists(json_path):
        os.remove(json_path)
        logger.info(f"Removed json => {json_path}")

    data = load_project_data(project)
    return data, page

def on_project_select(project):
    """
    Ha projektet választanak, logger => project logs mappa,
    data betölt, page=1
    """
    if not project:
        logger.info("No project selected => empty data")
        return [], "", 1
    setup_project_logger(project)
    logger.info(f"Project selected => {project}")
    data = load_project_data(project)
    return data, project, 1

############################################
# GRADIO BLOCKS
############################################

with gr.Blocks(css=".overlong { background-color: #ffcccc !important; }") as demo:
    gr.Markdown("# AI Dubbing Chunk Checker App")

    data_state = gr.State([])
    project_state = gr.State("")
    page_state = gr.State(1)

    # Row => Project dropdown + Load Project + pages_info (Page X / Y)
    with gr.Row():
        # Választható projekt
        workdir_path = config["DIRECTORIES"]["workdir"]
        projects = [
            d for d in os.listdir(workdir_path)
            if os.path.isdir(os.path.join(workdir_path, d))
        ]
        projects.sort()

        initial_project = default_project if (default_project in projects) else None

        project_dropdown = gr.Dropdown(
            choices=projects,
            label="Select Project Directory",
            value=initial_project
        )
        load_button = gr.Button("Load Project")
        # Oldalszámlálás (Markdown), a "Page X / Y" jelenik meg
        pages_info = gr.Markdown("Page 1 / 1")

    # FELSŐ GOMBOK
    with gr.Row():
        prev_button_top = gr.Button("Previous")
        next_button_top = gr.Button("Next")

    # TÁBLÁZAT (10 sor × 6 mező)
    base_boxes = []
    splits_audio_boxes = []
    splits_text_boxes = []
    trans_audio_boxes = []
    trans_text_boxes = []
    overlong_boxes = []
    row_slots = []

    for i in range(ROW_COUNT):
        with gr.Row() as row:
            r_basename = gr.Textbox(label="Basename", interactive=False)
            r_splits_audio = gr.Audio(label="Splits Audio", type="filepath")
            r_splits_text = gr.Textbox(label="Splits Text", interactive=False, lines=3)
            with gr.Column():
                r_trans_audio = gr.Audio(label="Translated Audio", type="filepath")
                r_delete = gr.Button("Delete Audio")
            with gr.Column():
                r_trans_text = gr.Textbox(label="Translated Text", lines=3)
                r_save = gr.Button("Save Changes")
            r_overlong = gr.Textbox(label="Overlong Info", interactive=False, lines=1)

        row_slots.append(row)
        base_boxes.append(r_basename)
        splits_audio_boxes.append(r_splits_audio)
        splits_text_boxes.append(r_splits_text)
        trans_audio_boxes.append(r_trans_audio)
        trans_text_boxes.append(r_trans_text)
        overlong_boxes.append(r_overlong)

        # Delete
        def delete_callback(row_idx, proj, d_list, pg):
            real_idx = (pg - 1) * ROW_COUNT + row_idx
            if real_idx < len(d_list):
                basename = d_list[real_idx]["basename"]
                return delete_audio_and_refresh(proj, basename, pg)
            return d_list, pg

        r_delete.click(
            fn=lambda p, data_list, page, i=i: delete_callback(i, p, data_list, page),
            inputs=[project_state, data_state, page_state],
            outputs=[data_state, page_state]
        )

        # Save
        def save_callback(row_idx, text, proj, d_list, pg):
            real_idx = (pg - 1) * ROW_COUNT + row_idx
            if real_idx < len(d_list):
                basename = d_list[real_idx]["basename"]
                return save_text_and_refresh(proj, basename, text, pg)
            return d_list, pg

        r_save.click(
            fn=lambda p, txt, data_list, page, i=i: save_callback(i, txt, p, data_list, page),
            inputs=[project_state, r_trans_text, data_state, page_state],
            outputs=[data_state, page_state]
        )

    table_container = gr.Column(*row_slots)

    # ALSÓ GOMBOK
    with gr.Row():
        prev_button_bottom = gr.Button("Previous")
        next_button_bottom = gr.Button("Next")

    ######################################
    # refresh_table => 10×6 = 60 + 1 => pages_info
    ######################################
    def refresh_table(data, page):
        out_vals = []
        start_idx = (page - 1) * ROW_COUNT
        total = len(data)

        for i in range(ROW_COUNT):
            idx = start_idx + i
            if idx < total:
                row_data = data[idx]
                bn = row_data["basename"]
                s_audio = row_data["splits_wav"]
                s_text = row_data["splits_txt"]
                t_audio = row_data["trans_wav"]
                if t_audio and not os.path.exists(t_audio):
                    t_audio = None
                t_text = row_data["trans_txt"]
                ov_info = ""
                if row_data.get("overlong"):
                    ov_info = f"Overlong by {row_data['diff']:.2f} sec"
            else:
                bn = ""
                s_audio = None
                s_text = ""
                t_audio = None
                t_text = ""
                ov_info = ""

            out_vals.extend([bn, s_audio, s_text, t_audio, t_text, ov_info])

        # Kiírjuk a logba, és generálunk "Page X / Y" szöveget
        logger.info(f"Refreshed table => page={page}, total={total}")
        total_pages = math.ceil(total / ROW_COUNT) if total else 1
        page_text = f"Page {page} / {total_pages}"

        # Visszaadjuk a 60 table mező + a pages_info
        return tuple(out_vals) + (page_text,)

    # 60 mező + 1 => 61
    all_outputs = []
    for i in range(ROW_COUNT):
        all_outputs.append(base_boxes[i])
        all_outputs.append(splits_audio_boxes[i])
        all_outputs.append(splits_text_boxes[i])
        all_outputs.append(trans_audio_boxes[i])
        all_outputs.append(trans_text_boxes[i])
        all_outputs.append(overlong_boxes[i])
    # plusz a pages_info
    all_outputs.append(pages_info)

    # Ha data_state vagy page_state változik => refresh_table
    data_state.change(
        fn=refresh_table,
        inputs=[data_state, page_state],
        outputs=all_outputs
    )
    page_state.change(
        fn=refresh_table,
        inputs=[data_state, page_state],
        outputs=all_outputs
    )

    def next_page_func(d_list, pg, proj):
        total_pages = math.ceil(len(d_list)/ROW_COUNT) if d_list else 1
        new_pg = pg + 1 if pg < total_pages else pg
        return (d_list, new_pg)

    def prev_page_func(d_list, pg, proj):
        new_pg = pg - 1 if pg > 1 else pg
        return (d_list, new_pg)

    # Felső
    next_button_top.click(
        fn=next_page_func,
        inputs=[data_state, page_state, project_state],
        outputs=[data_state, page_state]
    )
    prev_button_top.click(
        fn=prev_page_func,
        inputs=[data_state, page_state, project_state],
        outputs=[data_state, page_state]
    )

    # Alsó
    next_button_bottom.click(
        fn=next_page_func,
        inputs=[data_state, page_state, project_state],
        outputs=[data_state, page_state]
    )
    prev_button_bottom.click(
        fn=prev_page_func,
        inputs=[data_state, page_state, project_state],
        outputs=[data_state, page_state]
    )

    def on_project_select_wrapper(proj):
        # Eredeti: data, project, page
        d, p, pg = on_project_select(proj)
        return d, p, pg

    project_dropdown.change(
        fn=on_project_select_wrapper,
        inputs=[project_dropdown],
        outputs=[data_state, project_state, page_state]
    )
    load_button.click(
        fn=on_project_select_wrapper,
        inputs=[project_dropdown],
        outputs=[data_state, project_state, page_state]
    )

    gr.Markdown("## File Table")
    table_container

# Indítás: share=True és host beállítása "0.0.0.0"-ra
demo.launch(share=True, server_name="0.0.0.0")
