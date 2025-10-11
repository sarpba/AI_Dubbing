#!/usr/bin/env python3
import argparse
import re

def parse_srt(file_path):
    """
    Egy SRT fájlt blokkok listájává alakít.
    Minden blokk egy tuple: (index, time_line, subtitle_text)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # A blokkokat egy vagy több üres sor választja el
    blocks = re.split(r'\n\s*\n', content)
    parsed_blocks = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 2:
            index = lines[0].strip()
            time_line = lines[1].strip()
            subtitle_text = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
            parsed_blocks.append((index, time_line, subtitle_text))
    return parsed_blocks

def write_srt(blocks, output_file):
    """
    Egy SRT blokkok listáját írja ki a fájlba.
    Az indexek 1-től kezdődnek.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (_, time_line, subtitle_text) in enumerate(blocks, start=1):
            f.write(f"{i}\n")
            f.write(f"{time_line}\n")
            if subtitle_text:
                f.write(f"{subtitle_text}\n")
            f.write("\n")

def parse_time(time_str):
    """
    Egy időpontot alakít át másodpercekben mérhető számmá.
    Várható formátum: HH:MM:SS,mmm
    """
    hours, minutes, sec_ms = time_str.split(':')
    seconds, ms = sec_ms.split(',')
    total = int(hours)*3600 + int(minutes)*60 + int(seconds) + int(ms)/1000
    return total

def parse_time_interval(time_line):
    """
    Egy időintervallumot dolgoz fel, pl.:
    "00:01:48,692 --> 00:01:49,901"
    Visszaadja: (start_time, end_time)
    """
    parts = time_line.split('-->')
    if len(parts) != 2:
        raise ValueError(f"Hibás időintervallum formátum: {time_line}")
    start_str = parts[0].strip()
    end_str = parts[1].strip()
    return parse_time(start_str), parse_time(end_str)

def main():
    parser = argparse.ArgumentParser(
        description="Az angol SRT fájlt (-eng) alapul véve kikeresi a magyar feliratokat (-hun) úgy, "
                    "hogy ha van pontos időegyezés vagy a magyar blokk több angol időintervallumot fed le, "
                    "a magyar szöveget sorokra bontva hozzárendeli az angol időzítésekhez. "
                    "Azokat az angol blokkokat, ahol nincs megfelelő magyar felirat, változatlanul írja ki."
    )
    parser.add_argument('-eng', required=True, help="Angol SRT fájl elérési útvonala")
    parser.add_argument('-hun', required=True, help="Magyar SRT fájl elérési útvonala")
    parser.add_argument('-o', required=True, help="Kimeneti SRT fájl elérési útvonala")
    args = parser.parse_args()

    # Angol feliratok feldolgozása
    eng_blocks = parse_srt(args.eng)
    eng_data = []
    for block in eng_blocks:
        idx, time_line, text = block
        try:
            start, end = parse_time_interval(time_line)
        except ValueError as e:
            print(f"Hiba az angol blokk (index {idx}) időintervallumának feldolgozásánál: {e}")
            continue
        eng_data.append({
            'orig_index': idx,
            'time_line': time_line,
            'start': start,
            'end': end,
            'text': text
        })

    # Egy mapping, amely az angol blokk indexéhez hozzárendeli a magyar szöveget (vagy annak részleteit)
    eng_assignment = {}

    # Magyar feliratok feldolgozása
    hun_blocks = parse_srt(args.hun)
    for h_block in hun_blocks:
        h_index, h_time_line, h_text = h_block
        try:
            h_start, h_end = parse_time_interval(h_time_line)
        except ValueError as e:
            print(f"Hiba a magyar blokk (index {h_index}) időintervallumának feldolgozásánál: {e}")
            continue

        # Először ellenőrizzük, hogy a magyar blokk időzítése pontosan megegyezik-e valamelyik angoléval
        if any(h_time_line == eng['time_line'] for eng in eng_data):
            for i, eng in enumerate(eng_data):
                if eng['time_line'] == h_time_line:
                    eng_assignment[i] = h_text
            continue

        # Ha nincs pontos egyezés, keressük azokat az angol blokkokat, amelyek teljesen beleesnek a magyar blokk intervallumába
        matching_eng_idxs = []
        for i, eng in enumerate(eng_data):
            if eng['start'] >= h_start and eng['end'] <= h_end:
                matching_eng_idxs.append(i)
        if not matching_eng_idxs:
            continue

        # A magyar blokkot sorokra bontjuk (üres sorok kihagyásával)
        h_lines = [line.strip() for line in h_text.splitlines() if line.strip()]
        # Ha a magyar sorok száma megegyezik a lefedett angol blokkok számával, hozzárendeljük soronként
        if len(h_lines) == len(matching_eng_idxs):
            for eng_idx, h_line in zip(matching_eng_idxs, h_lines):
                eng_assignment[eng_idx] = h_line
        # Egyéb esetben a blokkot kihagyjuk

    # Kimenet: minden angol blokkot feldolgozunk.
    # Ha egy angol blokkhoz található magyar megfelelés (az eng_assignment-ben szerepel), azt használjuk;
    # ha nem, akkor az eredeti angol szöveget írjuk ki.
    output_blocks = []
    for i, eng in enumerate(eng_data):
        if i in eng_assignment:
            out_text = eng_assignment[i]
        else:
            out_text = eng['text']
        output_blocks.append((eng['orig_index'], eng['time_line'], out_text))

    if not output_blocks:
        print("Nincs olyan felirat, amelynél megtaláltuk volna a megfelelő magyar szöveget, és az angol sem került kimenetre.")
    else:
        write_srt(output_blocks, args.o)
        print(f"A feliratok mentése megtörtént: {args.o}")

if __name__ == "__main__":
    main()
