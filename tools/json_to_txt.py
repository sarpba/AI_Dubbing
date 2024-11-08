import json
import argparse

# Argumentumok feldolgozása
parser = argparse.ArgumentParser(description='JSON segments text sorainak másolása egy TXT fájlba.')
parser.add_argument('-i', '--input', required=True, help='Input JSON fájl neve')
parser.add_argument('-o', '--output', required=True, help='Output TXT fájl neve')
args = parser.parse_args()

# JSON fájl betöltése
input_file = args.input
output_file = args.output

# JSON fájl megnyitása és olvasása
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Szövegek másolása egy TXT fájlba
with open(output_file, 'w', encoding='utf-8') as file:
    for segment in data['segments']:
        file.write(segment['text'] + '\n')

print(f'Szövegek másolása kész. Az új fájl neve: {output_file}')

