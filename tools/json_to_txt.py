import json
import argparse
from pathlib import Path

try:
    from tools.debug_utils import add_debug_argument, configure_debug_mode
except ImportError:  # pragma: no cover - allows running standalone
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from tools.debug_utils import add_debug_argument, configure_debug_mode


def main() -> None:
    parser = argparse.ArgumentParser(description='JSON segments text sorainak másolása egy TXT fájlba.')
    parser.add_argument('-i', '--input', required=True, help='Input JSON fájl neve')
    parser.add_argument('-o', '--output', required=True, help='Output TXT fájl neve')
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)

    # JSON fájl megnyitása és olvasása
    with open(args.input, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Szövegek másolása egy TXT fájlba
    with open(args.output, 'w', encoding='utf-8') as file:
        for segment in data['segments']:
            file.write(segment['text'] + '\n')

    print(f'Szövegek másolása kész. Az új fájl neve: {args.output}')


if __name__ == '__main__':
    main()
