import os
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from jiwer import wer, cer, Compose, RemovePunctuation, ToLowerCase, RemoveMultipleSpaces
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import librosa

def collate_fn(batch):
    return batch

def update_eval_csv(eval_csv_path, model_name, WER_val, CER_val, norm_WER_val, norm_CER_val, dataset_base, batch_size, language, runtime):
    # Ha már létezik a CSV, beolvassuk
    if os.path.exists(eval_csv_path):
        eval_df = pd.read_csv(eval_csv_path)
    else:
        eval_df = pd.DataFrame(columns=["model_name", "WER", "CER", "Norm WER", "Norm CER", "dataset", "batch_size", "language", "runtime"])
    
    # Ellenőrizzük, van-e már sor ugyanazzal a model_name + dataset kombinációval
    mask = (eval_df["model_name"] == model_name) & (eval_df["dataset"] == dataset_base)
    eval_df = eval_df[~mask]  # Töröljük az esetleg meglévő sort

    # Új sor hozzáadása
    new_row = {
        "model_name": model_name,
        "WER": WER_val,
        "CER": CER_val,
        "Norm WER": norm_WER_val,
        "Norm CER": norm_CER_val,
        "dataset": dataset_base,
        "batch_size": batch_size,
        "language": language,
        "runtime": runtime
    }
    eval_df = pd.concat([eval_df, pd.DataFrame([new_row])], ignore_index=True)

    # CSV mentése
    eval_df.to_csv(eval_csv_path, index=False)

    return eval_df

def create_markdown_from_eval(eval_df, eval_txt_path):
    # Rendezés Normalizált WER szerint
    eval_df_sorted = eval_df.sort_values(by="Norm WER", ascending=True)

    # Markdown táblázat készítése
    with open(eval_txt_path, "w", encoding="utf-8") as f:
        f.write("| model_name | WER | CER | Norm WER | Norm CER | dataset | batch_size | language | runtime |\n")
        f.write("|------------|-----|-----|-----------------|-----------------|----------|------------|----------|---------|\n")
        for _, row in eval_df_sorted.iterrows():
            f.write(
                f"| {row['model_name']} | {row['WER']:.2f} | {row['CER']:.2f} | {row['Norm WER']:.2f} | {row['Norm CER']:.2f} | {row['dataset']} | {row['batch_size']} | {row['language']} | {row['runtime']:.2f} |\n"
            )

def main():
    # Paraméterek beállítása
    model_names = [
    	#"openai/whisper-tiny",
    	#"openai/whisper-base",
	#"openai/whisper-small",
	#"openai/whisper-medium",
	#"openai/whisper-large",
	#"openai/whisper-large-v2",
	#"openai/whisper-large-v3",
	#"sarpba/whisper-hu-tiny-finetuned",
	#"sarpba/whisper-base-hungarian_v1",
	"sarpba/whisper-hu-small-finetuned",
    ]
    
    CSV_PATHS = [
        "/home/sarpba/audio_tests/CV_17_0_hu_test.csv",
        "/home/sarpba/audio_tests/g_fleurs_test_hu.csv",
    ]
    
    language = "hu"  # Nyelvkód a Whisper modellhez
    initial_batch_size = 32  # Batch mérete induláskor
    csv_file = "model_results.csv"  # CSV fájl neve az eredményekhez (per-model/per-dataset)
    max_duration_seconds = 30  # Maximális fájl hossz
    eval_csv_path = os.path.join("test", "eval.csv")
    eval_txt_path = os.path.join("test", "eval.txt")

    # Eszköz kiválasztása
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Használt eszköz: {device}")

    for model_name in model_names:
        print(f"\n=== Modell tesztelése: {model_name} ===")

        # Modell és processzor betöltése
        print("Modell és processzor betöltése...")
        processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print("Modell és processzor sikeresen betöltve.")

        for CSV_PATH in CSV_PATHS:
            start_time = time.time()

            csv_base = os.path.splitext(os.path.basename(CSV_PATH))[0]
            txt_file = f"{model_name.replace('/', '_')}_{csv_base}.txt"
            output_dir = os.path.join("test", model_name, csv_base)    
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n--- Adatkészlet tesztelése: {CSV_PATH} ---")

            # Adat betöltése helyi CSV-ből
            print("Adatkészlet betöltése helyi CSV fájlból...")
            data_files = {"train": CSV_PATH}
            raw_datasets = load_dataset("csv", data_files=data_files, sep="|", column_names=["audio", "text"], quoting=3)
            
            # Audio típusra alakítás, 16000Hz-re resample
            raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))
            
            # Adatfelosztás
            raw_datasets = raw_datasets["train"].train_test_split(test_size=0.99, seed=42)
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["test"]
            print("Adatkészlet sikeresen betöltve és felosztva.")

            reference_key = "text"

            # Függvény az audio hosszának szűrésére
            def filter_long_audio(example):
                audio = example['audio']
                duration = len(audio['array']) / audio['sampling_rate']
                return duration <= max_duration_seconds

            # Függvény a rövid vagy None transzkripciók szűrésére
            def filter_short_text(example):
                txt = example[reference_key]
                return (txt is not None) and (len(txt.strip()) >= 3)

            # Szűrés audio hossz alapján
            print(f"Szűrés audio fájlok hosszúsága alapján (max {max_duration_seconds} másodperc)...")
            initial_count = len(eval_dataset)
            eval_dataset = eval_dataset.filter(filter_long_audio)
            filtered_count_by_audio = len(eval_dataset)
            skipped_count_by_audio = initial_count - filtered_count_by_audio
            print(f"Összes eval audio fájl: {initial_count}")
            print(f"Kiszűrt eval audio fájlok (audio hossza alapján): {skipped_count_by_audio}")
            print(f"Feldolgozott eval audio fájlok (audio hossza alapján): {filtered_count_by_audio}")

            # Szűrés szövegek alapján
            initial_count_text = len(eval_dataset)
            eval_dataset = eval_dataset.filter(filter_short_text)
            filtered_count_text = len(eval_dataset)
            skipped_count_text = initial_count_text - filtered_count_text
            print(f"Kiszűrt eval audio fájlok (szöveg hossza alapján): {skipped_count_text}")
            print(f"Feldolgozott eval audio fájlok (szöveg hossza alapján): {filtered_count_text}")

            # Az alábbi ciklus megpróbálja lefuttatni a tesztet az aktuális batch_size mellett
            # Ha elfogy a memória, csökkenti a batch_size-t és újrapróbálja.
            batch_size = initial_batch_size
            results = []
            while True:
                try:
                    print(f"Próbálkozás batch_size = {batch_size}-val/vel...")
                    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                    # Normalizáció WER/CER-hez
                    normalization_transform = Compose([
                        ToLowerCase(),
                        RemovePunctuation(),
                        RemoveMultipleSpaces()
                    ])

                    for batch in tqdm(dataloader, desc="Feldolgozás"):
                        audios = [example['audio'] for example in batch]
                        references = [example[reference_key].strip() for example in batch]

                        # Ellenőrizzük a batch mintavételezési rátáit
                        sampling_rates = set(audio['sampling_rate'] for audio in audios)
                        if len(sampling_rates) != 1:
                            print("Figyelem: eltérő mintavételezési ráták egy batch-ben!")
                            continue
                        sampling_rate = audios[0]['sampling_rate']

                        # Audio átmeneti mintavételezése 16000 Hz-re
                        resampled_audios = [librosa.resample(audio["array"], orig_sr=sampling_rate, target_sr=16000) for audio in audios]

                        # Audio feldolgozása a processzorral
                        input_features = processor(
                            resampled_audios,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=True
                        )

                        input_features['input_features'] = input_features['input_features'].to(device)

                        # Pad vagy vágás a mel-spectrogramra
                        desired_length = 3000
                        current_length = input_features['input_features'].shape[-1]
                        if current_length < desired_length:
                            pad_length = desired_length - current_length
                            padding = torch.zeros(
                                input_features['input_features'].shape[0],
                                input_features['input_features'].shape[1],
                                pad_length
                            ).to(device)
                            input_features['input_features'] = torch.cat([input_features['input_features'], padding], dim=-1)
                        elif current_length > desired_length:
                            input_features['input_features'] = input_features['input_features'][:, :, :desired_length]

                        input_features['attention_mask'] = torch.ones_like(input_features['input_features']).to(device)
                        input_features = {k: v.to(device) for k, v in input_features.items()}

                        # Transzkripció generálása
                        with torch.no_grad():
                            generated_ids = model.generate(**input_features)
                            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

                        # Metrikák számítása
                        for transcription, reference, example in zip(transcriptions, references, batch):
                            transcription = transcription.strip()
                            reference = reference.strip()

                            current_wer = wer(reference, transcription)
                            normalized_reference = normalization_transform(reference)
                            normalized_transcription = normalization_transform(transcription)
                            normalized_wer = wer(normalized_reference, normalized_transcription)

                            current_cer = cer(reference, transcription)
                            normalized_cer = cer(normalized_reference, normalized_transcription)

                            results.append({
                                "transcription": transcription,
                                "reference": reference,
                                "WER": current_wer,
                                "CER": current_cer,
                                "Normalized_WER": normalized_wer,
                                "Normalized_CER": normalized_cer
                            })
                    # Ha idáig eljutottunk hiba nélkül, akkor kilépünk a while-ból
                    break

                except RuntimeError as e:
                    # Ha elfogy a memória, csökkentjük a batch_size-t
                    if "out of memory" in str(e).lower():
                        print(f"CUDA memóriaprobléma lépett fel batch_size={batch_size} mellett. Csökkentés...")
                        batch_size = batch_size // 2
                        if batch_size < 1:
                            print("Nem sikerült 1-es batch_size mellett sem futtatni a modellt. Kilépés.")
                            results = []
                            break
                        torch.cuda.empty_cache()
                        continue
                    else:
                        # Egyéb hibák továbbdobása
                        raise e

            if len(results) == 0:
                print("Nincs feldolgozott adat vagy nem sikerült futtatni.")
                continue

            df = pd.DataFrame(results)
            avg_wer = df["WER"].mean() * 100
            avg_cer = df["CER"].mean() * 100
            avg_normalized_wer = df["Normalized_WER"].mean() * 100
            avg_normalized_cer = df["Normalized_CER"].mean() * 100

            summary = {
                "Average_WER": avg_wer,
                "Average_CER": avg_cer,
                "Average_Normalized_WER": avg_normalized_wer,
                "Average_Normalized_CER": avg_normalized_cer
            }

            summary_df = pd.DataFrame([summary])
            full_df = pd.concat([df, summary_df], ignore_index=True)

            # CSV mentése (per-model/per-dataset)
            csv_path = os.path.join(output_dir, csv_file)
            full_df.to_csv(csv_path, index=False)
            print(f"Eredmények elmentve a {csv_path} fájlba.")

            runtime = time.time() - start_time

            # Összegző kiírás
            print("\n### Összesített Metrikák ###")
            print(f"WER: {avg_wer:.2f}%")
            print(f"CER: {avg_cer:.2f}%")
            print(f"Norm WER: {avg_normalized_wer:.2f}%")
            print(f"Norm CER: {avg_normalized_cer:.2f}%")

            # TXT fájl mentése (per-model/per-dataset)
            txt_path = os.path.join(output_dir, txt_file)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("### Összesített Metrikák ###\n")
                f.write(f"WER: {avg_wer:.2f}%\n")
                f.write(f"CER: {avg_cer:.2f}%\n")
                f.write(f"Norm WER: {avg_normalized_wer:.2f}%\n")
                f.write(f"Norm CER: {avg_normalized_cer:.2f}%\n\n")

                for result in results:
                    f.write(f"REF: {result['reference']}\n")
                    f.write(f"HYP: {result['transcription']}\n")
                    f.write("---\n")

            print(f"Összesített eredmények elmentve a {txt_path} fájlba.")

            # Közös eval.csv frissítése
            eval_df = update_eval_csv(
                eval_csv_path=eval_csv_path,
                model_name=model_name,
                WER_val=avg_wer,
                CER_val=avg_cer,
                norm_WER_val=avg_normalized_wer,
                norm_CER_val=avg_normalized_cer,
                dataset_base=csv_base,
                batch_size=batch_size,
                language=language,
                runtime=runtime
            )

            # Eval markdown generálása
            create_markdown_from_eval(eval_df, eval_txt_path)
            print(f"Markdown mentve: {eval_txt_path}")

if __name__ == "__main__":
    main()

