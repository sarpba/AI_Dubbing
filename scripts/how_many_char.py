import os
import argparse

def count_characters_in_directory(input_dir):
    total_characters = 0
    
    # Végigmegyünk az input_dir összes .txt fájlján
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_characters = len(content)
                total_characters += file_characters
                print(f"{filename} karakterek száma: {file_characters}")

    return total_characters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Számolja a .txt fájlok karaktereit egy könyvtárban')
    parser.add_argument('-i', '--input_dir', required=True, help='A könyvtár, amely tartalmazza a karakterek összeszámolásához szükséges .txt fájlokat')
    
    args = parser.parse_args()
    total_characters = count_characters_in_directory(args.input_dir)
    print(f"Összes karakterek száma: {total_characters}")

