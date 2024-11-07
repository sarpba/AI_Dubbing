import os
import shutil
import argparse
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Másolja a .txt fájlokat, amelyek számot, speciális karaktert vagy a "dr" szót tartalmaznak.')
    parser.add_argument('-i', '--input', required=True, help='Bemeneti könyvtár elérési útja')
    parser.add_argument('-o', '--output', required=True, help='Kimeneti könyvtár elérési útja')
    return parser.parse_args()

def contains_number_special_or_dr(filepath, pattern):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                if pattern.search(line):
                    return True
    except Exception as e:
        print(f"Hiba történt a fájl olvasása közben: {filepath}\n{e}")
    return False

def main():
    args = parse_arguments()
    input_dir = args.input
    output_dir = args.output

    # Ellenőrzi, hogy a bemeneti könyvtár létezik
    if not os.path.isdir(input_dir):
        print(f"A bemeneti könyvtár nem található: {input_dir}")
        return

    # Létrehozza a kimeneti könyvtárat, ha nem létezik
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Létrehozva a kimeneti könyvtár: {output_dir}")

    # Definiálja a keresendő mintát (szám, speciális karakterek vagy a "dr" szó)
    # A "dr" szó keresése szóhatárok között, kis- és nagybetűkre érzéketlenül
    pattern = re.compile(r'[0-9%#&\$˚]|\bdr\b', re.IGNORECASE)

    # Bejárja az input könyvtár .txt fájljait
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                input_file_path = os.path.join(root, file)
                if contains_number_special_or_dr(input_file_path, pattern):
                    # Meghatározza a relatív útvonalat az input könyvtárhoz képest
                    rel_path = os.path.relpath(root, input_dir)
                    # Célkönyvtár elérési útja
                    dest_dir = os.path.join(output_dir, rel_path)
                    # Létrehozza a célkönyvtárat, ha nem létezik
                    os.makedirs(dest_dir, exist_ok=True)
                    # Célfájl elérési útja
                    dest_file_path = os.path.join(dest_dir, file)
                    # Másolja a fájlt
                    shutil.copy2(input_file_path, dest_file_path)
                    print(f"Másolva: {input_file_path} -> {dest_file_path}")

if __name__ == "__main__":
    main()
