import os
import argparse
import json
from pydub import AudioSegment

def process_audio(input_dir, reference_json_dir, delete_empty):
    """
    Levágja vagy hozzáadja az audió fájlok elejéről a csendet az input és referencia JSON időbélyegek alapján,
    majd felülírja az eredeti fájlokat. Sikeres feldolgozás után törli az input JSON fájlt.
    Ha a JSON fájl üres és a --delete_empty kapcsoló be van kapcsolva,
    akkor törli az audió fájlt is.
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".mp3", ".wav")):
            audio_path = os.path.join(input_dir, filename)
            input_json_path = os.path.join(input_dir, os.path.splitext(filename)[0] + '.json')
            reference_json_path = os.path.join(reference_json_dir, os.path.splitext(filename)[0] + '.json')

            print(f"\nFeldolgozás indul: {filename}")
            print(f"Input JSON fájl elérhetősége: {input_json_path}")
            print(f"Referencia JSON fájl elérhetősége: {reference_json_path}")

            # Ellenőrizze, hogy mind az input, mind a referencia JSON fájl létezik
            if not os.path.exists(input_json_path):
                print(f"Input JSON fájl nem található a {filename} fájlhoz, kihagyás.")
                continue

            if not os.path.exists(reference_json_path):
                print(f"Referencia JSON fájl nem található a {filename} fájlhoz, kihagyás.")
                if delete_empty:
                    try:
                        os.remove(input_json_path)
                        os.remove(audio_path)
                        print(f"Törölve: {input_json_path} és {audio_path}.")
                    except OSError as e:
                        print(f"Hiba a fájlok törlésekor: {e}")
                continue

            # JSON adatok betöltése
            try:
                with open(input_json_path, 'r') as f:
                    input_data = json.load(f)
                with open(reference_json_path, 'r') as f:
                    reference_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Érvénytelen JSON formátum a {filename} fájlhoz tartozó JSON-ban, kihagyás.")
                continue

            # Ellenőrizze, hogy mindkét JSON fájlban van-e legalább egy szegmens
            input_segments = input_data.get("segments", [])
            reference_segments = reference_data.get("segments", [])

            if not input_segments or not reference_segments:
                print(f"Hiányzik szegmens a {filename} fájlhoz tartozó JSON-ban.")
                if delete_empty:
                    try:
                        os.remove(input_json_path)
                        os.remove(audio_path)
                        print(f"Törölve: {input_json_path} és {audio_path}.")
                    except OSError as e:
                        print(f"Hiba a fájlok törlésekor: {e}")
                continue

            # Az első szegmens kezdési idejének meghatározása
            input_start_time = input_segments[0].get("start", 0)
            reference_start_time = reference_segments[0].get("start", 0)

            print(f"Input JSON első timestampje: {input_start_time:.3f} másodperc")
            print(f"Referencia JSON első timestampje: {reference_start_time:.3f} másodperc")

            # Különbség kiszámítása
            time_difference = reference_start_time - input_start_time  # Pozitív érték: hozzáadás, Negatív: további levágás

            if time_difference > 0:
                action = "hozzáadás"
            elif time_difference < 0:
                action = "levágás"
            else:
                action = "nincs módosítás"

            print(f"Különbség: {time_difference:.3f} másodperc - Művelet: {action}")

            # Audió betöltése
            try:
                audio = AudioSegment.from_file(audio_path)
            except Exception as e:
                print(f"Hiba a {filename} audió betöltése során: {e}")
                continue

            if time_difference > 0:
                # Hozzáadás: Csendet kell hozzáadni az audió elejéhez
                silence_duration = time_difference * 1000  # Átváltás milliszekundumra
                silence = AudioSegment.silent(duration=silence_duration)
                adjusted_audio = silence + audio
                print(f"Hozzáadás: {silence_duration / 1000:.3f} másodperc csend a {filename} elejéhez.")
            elif time_difference < 0:
                # Levágás: További csendet kell levágni az audió elejéről
                trim_duration = abs(time_difference) * 1000  # Átváltás milliszekundumra
                if trim_duration >= len(audio):
                    print(f"A levágás ({trim_duration / 1000:.3f} másodperc) nagyobb vagy egyenlő az audió hosszával ({len(audio) / 1000:.3f} másodperc) a {filename} fájlban. Törlés.")
                    if delete_empty:
                        try:
                            os.remove(input_json_path)
                            os.remove(audio_path)
                            print(f"Törölve: {input_json_path} és {audio_path}.")
                        except OSError as e:
                            print(f"Hiba a fájlok törlésekor: {e}")
                    continue
                adjusted_audio = audio[trim_duration:]
                print(f"Levágás: {trim_duration / 1000:.3f} másodperc a {filename} elejéről.")
            else:
                # Nincs szükség módosításra
                adjusted_audio = audio
                print(f"Nincs szükség módosításra a {filename} fájlban.")

            # Az eredeti audió felülírása a módosított audióval
            try:
                adjusted_audio.export(audio_path, format="mp3" if filename.lower().endswith(".mp3") else "wav")
                print(f"Feldolgozva: {filename}.")
            except Exception as e:
                print(f"Hiba a {filename} levágott audió exportálásakor: {e}")
                continue

            # JSON fájl törlése a feldolgozás után
            try:
                os.remove(input_json_path)
                print(f"Törölve: {input_json_path} a jövőbeni újrafeldolgozás érdekében.")
            except OSError as e:
                print(f"Hiba a {input_json_path} törlésekor: {e}")

def synchronize_loudness(sync_dir, reference_dir, min_db):
    """
    Szinkronizálja a 'sync' könyvtárban lévő audió fájlok hangerőszintjét a referencia
    audió fájlok maximális hangerőszintjéhez. Csak azoknál a fájloknál végzi el a
    szinkronizálást, ahol a referencia audió maximális hangerőszintje >= min_db.
    """
    if not os.path.exists(sync_dir):
        print(f"A szinkronizációs könyvtár {sync_dir} nem létezik. A hangerő szinkronizálás kihagyva.")
        return

    if not os.path.exists(reference_dir):
        print(f"A referencia könyvtár {reference_dir} nem létezik. A hangerő szinkronizálás kihagyva.")
        return

    for sync_filename in os.listdir(sync_dir):
        if not sync_filename.lower().endswith(".wav"):
            continue  # Csak WAV fájlok feldolgozása a sync könyvtárban

        sync_path = os.path.join(sync_dir, sync_filename)
        base_name = os.path.splitext(sync_filename)[0]

        print(f"\nHangerő szinkronizálás indul: {sync_filename}")
        print(f"Sync audió fájl elérhetősége: {sync_path}")

        # Megfelelő referencia fájl keresése (.wav vagy .mp3)
        reference_path_wav = os.path.join(reference_dir, base_name + '.wav')
        reference_path_mp3 = os.path.join(reference_dir, base_name + '.mp3')

        if os.path.exists(reference_path_wav):
            reference_path = reference_path_wav
        elif os.path.exists(reference_path_mp3):
            reference_path = reference_path_mp3
        else:
            print(f"Nem található megfelelő referencia fájl a {sync_filename} számára, kihagyás.")
            continue

        print(f"Referencia audió fájl elérhetősége: {reference_path}")

        try:
            # Referencia audió betöltése
            reference_audio = AudioSegment.from_file(reference_path)
            # Maximális hangerőszint mérés
            reference_peak = reference_audio.max_dBFS
            print(f"Referencia audió maximális hangerőszintje: {reference_peak:.2f} dB")

            # Ellenőrzés, hogy a referencia hangerőszint >= min_db
            if reference_peak < min_db:
                print(f"A referencia audió {os.path.basename(reference_path)} maximális hangerőszintje ({reference_peak:.2f} dB) < {min_db} dB. Kihagyás.")
                continue

            # Sync audió betöltése
            sync_audio = AudioSegment.from_file(sync_path)
            sync_peak = sync_audio.max_dBFS
            print(f"Sync audió jelenlegi maximális hangerőszintje: {sync_peak:.2f} dB")

            # Gain kiszámítása a szinkronizáláshoz
            gain = reference_peak - sync_peak
            if gain == 0:
                print(f"Nincs hangerő beállítás szükséges a {sync_filename} fájlhoz.")
                continue

            # Gain alkalmazása
            adjusted_sync_audio = sync_audio.apply_gain(gain)
            print(f"Alkalmazott gain: {gain:.2f} dB a {sync_filename} fájlhoz.")

            # Módosított audió exportálása vissza a sync könyvtárba
            adjusted_sync_audio.export(sync_path, format="wav")
            print(f"Hangerő szinkronizálva: {sync_filename}.")

        except Exception as e:
            print(f"Hiba a hangerő szinkronizálása során a {sync_filename} fájlhoz: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Levágja vagy hozzáadja az audió fájlok csendjét JSON időbélyegek alapján és szinkronizálja a hangerőt referencia audiók alapján.")
    parser.add_argument("-i", "--input_dir", required=True, help="Könyvtár az audió és JSON fájlok számára, amelyeket levágni vagy módosítani szeretnél.")
    parser.add_argument("-rj", "--reference_json_dir", required=True, help="Könyvtár az eredeti (referencia) JSON fájlok számára.")
    parser.add_argument("--delete_empty", action="store_true", help="Törölje az audió és JSON fájlokat időbélyeg hiányában.")
    parser.add_argument("--ira", help="Referencia audió fájlok könyvtára a hangerő szinkronizáláshoz.")
    parser.add_argument("-db", "--min_db", type=float, default=-40.0, help="Minimális referencia hangerőszint (dB) a szinkronizáláshoz. Alapértelmezett: -40.0 dB.")

    args = parser.parse_args()

    # Audió fájlok levágása vagy csend hozzáadása, valamint JSON kezelése
    process_audio(args.input_dir, args.reference_json_dir, args.delete_empty)

    # Hangerő szinkronizálás végrehajtása, ha --ira meg van adva
    if args.ira:
        # Sync könyvtár meghatározása
        project_dir = os.path.dirname(os.path.abspath(args.ira))
        sync_dir = os.path.join(project_dir, "sync")

        print("\nHangerő szinkronizálás elkezdése...")
        synchronize_loudness(sync_dir, args.ira, args.min_db)
        print("Hangerő szinkronizálás befejezve.")

