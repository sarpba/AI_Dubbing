import os
import subprocess
from multiprocessing import Pool
import argparse

# Konfigurációs változók
INFERENCE_SCRIPT = "f5-tts_infer-cli"
#CONFIG_FILE = "/home/sarpba/F5-TTS_old3/inference-cli.toml"
MODEL = "F5-TTS"
CKPT = "/home/sarpba/AI_Sync/TTS/model_270000_hun.pt"
VOCAB = "/home/sarpba/AI_Sync/TTS/vocab.txt"
def get_num_gpus():
    """Lekéri a rendelkezésre álló GPU-k számát."""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, check=True, text=True)
        gpus = result.stdout.strip().split('\n')
        return len(gpus)
    except Exception as e:
        print("Nem sikerült meghatározni a GPU-k számát. Alapértelmezett: 1 GPU.")
        return 1

def process_task(task, gpu_id):
    """Egyetlen feladat feldolgozása egy meghatározott GPU-val."""
    filename, paths, config = task
    basename = os.path.splitext(filename)[0]
    ref_audio_orig = os.path.join(paths['REF_DIR'], filename)
    ref_text_file = os.path.join(paths['REF_DIR'], f"{basename}.txt")
    gen_text_file = os.path.join(paths['GEN_TEXT_DIR'], f"{basename}.txt")

    output_wav = os.path.join(paths['OUTPUT_DIR'], f"{basename}.wav")

    if os.path.isfile(ref_text_file) and os.path.isfile(gen_text_file):
        # Ellenőrizzük, hogy a kimeneti fájl már létezik-e
        if os.path.exists(output_wav):
            print(f"A fájl már feldolgozva van és létezik: {output_wav}. Átugrás.")
            return

        # Létrehozunk egy egyedi almappát a kimeneti fájl számára
        unique_output_subdir = os.path.join(paths['OUTPUT_DIR'], f"{basename}_temp")
        os.makedirs(unique_output_subdir, exist_ok=True)

        with open(ref_text_file, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()
        with open(gen_text_file, 'r', encoding='utf-8') as f:
            gen_text = '... ' + f.read().strip().lower()  # Előtag hozzáadása

        ext = os.path.splitext(filename)[1].lower()
        if ext == '.mp3':
            # Konvertáljuk az mp3 fájlt wav formátumra
            ref_audio = os.path.join(paths['REF_DIR'], f"{basename}.wav")
            command = ['ffmpeg', '-y', '-i', ref_audio_orig, ref_audio]
            try:
                subprocess.run(command, check=True)
                print(f"Konvertálva: {ref_audio_orig} -> {ref_audio}")
            except subprocess.CalledProcessError as e:
                print(f"Hiba történt az mp3 konvertálása során: {e}")
                return
        elif ext == '.wav':
            ref_audio = ref_audio_orig
        else:
            print(f"Nem támogatott audio fájl formátum: {ext} a(z) {filename} fájlban.")
            return

        command = [
            INFERENCE_SCRIPT, #'python', INFERENCE_SCRIPT,
            #'--config', CONFIG_FILE,
            '--model', MODEL,
            '-p', CKPT,
            '-v', VOCAB,
            '--ref_audio', ref_audio,
            '--ref_text', ref_text,
            '--gen_text', gen_text,
            '--output_dir', unique_output_subdir
        ]

        # Beállítjuk a CUDA_VISIBLE_DEVICES környezeti változót a megfelelő GPU használatához
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            subprocess.run(command, check=True, env=env)
            print(f"Sikeresen feldolgozva: {basename} (GPU {gpu_id})")

            # Átnevezzük az out.wav fájlt a bemeneti fájl nevére
            temp_output_wav = os.path.join(unique_output_subdir, 'infer_cli_out.wav')
            if os.path.exists(temp_output_wav):
                os.rename(temp_output_wav, output_wav)
                print(f"Átnevezve: {temp_output_wav} -> {output_wav}")
                # Töröljük az ideiglenes almappát
                os.rmdir(unique_output_subdir)
            else:
                print(f"Nem található a kimeneti fájl: {temp_output_wav}")

        except subprocess.CalledProcessError as e:
            print(f"Hiba történt a(z) {basename} feldolgozása során: {e}")

        finally:
            # Töröljük a konvertált wav fájlt, ha mp3-ból konvertáltunk
            if ext == '.mp3':
                try:
                    os.remove(ref_audio)
                    print(f"Az ideiglenes wav fájl törölve: {ref_audio}")
                except OSError as e:
                    print(f"Hiba történt az ideiglenes wav fájl törlése során: {e}")
    else:
        print(f"Hiányzó szövegfájl(ok) a(z) {basename} fájlhoz.")

def main():
    """A fő függvény, amely elindítja a feldolgozást párhuzamosan több GPU-val."""
    parser = argparse.ArgumentParser(description="Audio fájlok feldolgozása több GPU-val.")
    parser.add_argument('-o', '--output_dir', required=True, help='A generált fájlok kimeneti könyvtára.')
    parser.add_argument('-r', '--ref_dir', required=True, help='Referencia audio és szöveg könyvtára.')
    parser.add_argument('-g', '--gen_text_dir', required=True, help='A generált szövegek könyvtára.')

    args = parser.parse_args()

    # Beállítjuk a könyvtárakat az argumentumok alapján
    REF_DIR = args.ref_dir
    OUTPUT_DIR = args.output_dir
    GEN_TEXT_DIR = args.gen_text_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_gpus = get_num_gpus()
    print(f"Talált GPU-k száma: {num_gpus}")

    # Feladatok gyűjtése
    tasks = []
    for filename in os.listdir(REF_DIR):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            basename = os.path.splitext(filename)[0]
            ref_text_file = os.path.join(REF_DIR, f"{basename}.txt")
            gen_text_file = os.path.join(GEN_TEXT_DIR, f"{basename}.txt")
            output_wav = os.path.join(OUTPUT_DIR, f"{basename}.wav")
            if os.path.isfile(ref_text_file) and os.path.isfile(gen_text_file):
                if os.path.exists(output_wav):
                    print(f"A fájl már feldolgozva van és létezik: {output_wav}. Átugrás.")
                    continue  # Átugorjuk a feldolgozást
                tasks.append((
                    filename,
                    {
                        'REF_DIR': REF_DIR,
                        'GEN_TEXT_DIR': GEN_TEXT_DIR,
                        'OUTPUT_DIR': OUTPUT_DIR
                    },
                    {
                        'INFERENCE_SCRIPT': INFERENCE_SCRIPT,
                        #'CONFIG_FILE': CONFIG_FILE,
                        'MODEL': MODEL
                    }
                ))
            else:
                print(f"Hiányzó szövegfájl(ok) a(z) {basename} fájlhoz.")
        else:
            print(f"Nem támogatott fájlformátum: {filename}")

    if not tasks:
        print("Nincs feldolgozandó feladat.")
        return

    # Feladatokhoz GPU-k hozzárendelése
    tasks_with_gpu = [ (task, idx % num_gpus) for idx, task in enumerate(tasks) ]

    # Pool létrehozása és a feladatok elosztása
    with Pool(processes=num_gpus) as pool:
        pool.starmap(process_task, tasks_with_gpu)

if __name__ == "__main__":
    main()
import os
import subprocess
from multiprocessing import Pool
import argparse

# Konfigurációs változók
INFERENCE_SCRIPT = "f5_tts_infer_cli"
#CONFIG_FILE = "/home/sarpba/F5-TTS_old3/inference-cli.toml"
MODEL = "F5-TTS"
CKPT = "hf://sarpba/F5-TTS-Hun/model_270000_hun_v3.pt"

def get_num_gpus():
    """Lekéri a rendelkezésre álló GPU-k számát."""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, check=True, text=True)
        gpus = result.stdout.strip().split('\n')
        return len(gpus)
    except Exception as e:
        print("Nem sikerült meghatározni a GPU-k számát. Alapértelmezett: 1 GPU.")
        return 1

def process_task(task, gpu_id):
    """Egyetlen feladat feldolgozása egy meghatározott GPU-val."""
    filename, paths, config = task
    basename = os.path.splitext(filename)[0]
    ref_audio_orig = os.path.join(paths['REF_DIR'], filename)
    ref_text_file = os.path.join(paths['REF_DIR'], f"{basename}.txt")
    gen_text_file = os.path.join(paths['GEN_TEXT_DIR'], f"{basename}.txt")

    output_wav = os.path.join(paths['OUTPUT_DIR'], f"{basename}.wav")

    if os.path.isfile(ref_text_file) and os.path.isfile(gen_text_file):
        # Ellenőrizzük, hogy a kimeneti fájl már létezik-e
        if os.path.exists(output_wav):
            print(f"A fájl már feldolgozva van és létezik: {output_wav}. Átugrás.")
            return

        # Létrehozunk egy egyedi almappát a kimeneti fájl számára
        unique_output_subdir = os.path.join(paths['OUTPUT_DIR'], f"{basename}_temp")
        os.makedirs(unique_output_subdir, exist_ok=True)

        with open(ref_text_file, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()
        with open(gen_text_file, 'r', encoding='utf-8') as f:
            gen_text = '... ' + f.read().strip().lower()  # Előtag hozzáadása

        ext = os.path.splitext(filename)[1].lower()
        if ext == '.mp3':
            # Konvertáljuk az mp3 fájlt wav formátumra
            ref_audio = os.path.join(paths['REF_DIR'], f"{basename}.wav")
            command = ['ffmpeg', '-y', '-i', ref_audio_orig, ref_audio]
            try:
                subprocess.run(command, check=True)
                print(f"Konvertálva: {ref_audio_orig} -> {ref_audio}")
            except subprocess.CalledProcessError as e:
                print(f"Hiba történt az mp3 konvertálása során: {e}")
                return
        elif ext == '.wav':
            ref_audio = ref_audio_orig
        else:
            print(f"Nem támogatott audio fájl formátum: {ext} a(z) {filename} fájlban.")
            return

        command = [
            'python', INFERENCE_SCRIPT,
            #'--config', CONFIG_FILE,
            '--model', MODEL,
            '-p', CKPT,
            '--ref_audio', ref_audio,
            '--ref_text', ref_text,
            '--gen_text', gen_text,
            '--output_dir', unique_output_subdir
        ]

        # Beállítjuk a CUDA_VISIBLE_DEVICES környezeti változót a megfelelő GPU használatához
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            subprocess.run(command, check=True, env=env)
            print(f"Sikeresen feldolgozva: {basename} (GPU {gpu_id})")

            # Átnevezzük az out.wav fájlt a bemeneti fájl nevére
            temp_output_wav = os.path.join(unique_output_subdir, 'out.wav')
            if os.path.exists(temp_output_wav):
                os.rename(temp_output_wav, output_wav)
                print(f"Átnevezve: {temp_output_wav} -> {output_wav}")
                # Töröljük az ideiglenes almappát
                os.rmdir(unique_output_subdir)
            else:
                print(f"Nem található a kimeneti fájl: {temp_output_wav}")

        except subprocess.CalledProcessError as e:
            print(f"Hiba történt a(z) {basename} feldolgozása során: {e}")

        finally:
            # Töröljük a konvertált wav fájlt, ha mp3-ból konvertáltunk
            if ext == '.mp3':
                try:
                    os.remove(ref_audio)
                    print(f"Az ideiglenes wav fájl törölve: {ref_audio}")
                except OSError as e:
                    print(f"Hiba történt az ideiglenes wav fájl törlése során: {e}")
    else:
        print(f"Hiányzó szövegfájl(ok) a(z) {basename} fájlhoz.")

def main():
    """A fő függvény, amely elindítja a feldolgozást párhuzamosan több GPU-val."""
    parser = argparse.ArgumentParser(description="Audio fájlok feldolgozása több GPU-val.")
    parser.add_argument('-o', '--output_dir', required=True, help='A generált fájlok kimeneti könyvtára.')
    parser.add_argument('-r', '--ref_dir', required=True, help='Referencia audio és szöveg könyvtára.')
    parser.add_argument('-g', '--gen_text_dir', required=True, help='A generált szövegek könyvtára.')

    args = parser.parse_args()

    # Beállítjuk a könyvtárakat az argumentumok alapján
    REF_DIR = args.ref_dir
    OUTPUT_DIR = args.output_dir
    GEN_TEXT_DIR = args.gen_text_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_gpus = get_num_gpus()
    print(f"Talált GPU-k száma: {num_gpus}")

    # Feladatok gyűjtése
    tasks = []
    for filename in os.listdir(REF_DIR):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            basename = os.path.splitext(filename)[0]
            ref_text_file = os.path.join(REF_DIR, f"{basename}.txt")
            gen_text_file = os.path.join(GEN_TEXT_DIR, f"{basename}.txt")
            output_wav = os.path.join(OUTPUT_DIR, f"{basename}.wav")
            if os.path.isfile(ref_text_file) and os.path.isfile(gen_text_file):
                if os.path.exists(output_wav):
                    print(f"A fájl már feldolgozva van és létezik: {output_wav}. Átugrás.")
                    continue  # Átugorjuk a feldolgozást
                tasks.append((
                    filename,
                    {
                        'REF_DIR': REF_DIR,
                        'GEN_TEXT_DIR': GEN_TEXT_DIR,
                        'OUTPUT_DIR': OUTPUT_DIR
                    },
                    {
                        'INFERENCE_SCRIPT': INFERENCE_SCRIPT,
                        'CONFIG_FILE': CONFIG_FILE,
                        'MODEL': MODEL
                    }
                ))
            else:
                print(f"Hiányzó szövegfájl(ok) a(z) {basename} fájlhoz.")
        else:
            print(f"Nem támogatott fájlformátum: {filename}")

    if not tasks:
        print("Nincs feldolgozandó feladat.")
        return

    # Feladatokhoz GPU-k hozzárendelése
    tasks_with_gpu = [ (task, idx % num_gpus) for idx, task in enumerate(tasks) ]

    # Pool létrehozása és a feladatok elosztása
    with Pool(processes=num_gpus) as pool:
        pool.starmap(process_task, tasks_with_gpu)

if __name__ == "__main__":
    main()

