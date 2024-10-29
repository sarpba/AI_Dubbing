import re
import csv

class SimpleHungarianTextNormalizer:
    def __init__(self, custom_replacements_file=None):
        # Egyszerű számok szöveges megfelelői 0-tól 99-ig
        self.units = ["nulla", "egy", "kettő", "három", "négy", "öt", "hat", "hét", "nyolc", "kilenc"]
        self.teens = ["tíz", "tizenegy", "tizenkettő", "tizenhárom", "tizennégy",
                     "tizenöt", "tizenhat", "tizenhét", "tizennyolc", "tizenkilencedik"]
        self.tens = ["", "", "húsz", "harminc", "negyven", "ötven",
                    "hatvan", "hetven", "nyolcvan", "kilencven"]
        
        # Egyéni cserék betöltése
        self.custom_replacements = {}
        if custom_replacements_file:
            self.load_custom_replacements(custom_replacements_file)

    def load_custom_replacements(self, filepath):
        """Betölti a szó-pár cseréket egy külső CSV fájlból."""
        try:
            with open(filepath, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    pattern = row['pattern'].lower()
                    replacement = row['replacement']
                    self.custom_replacements[pattern] = replacement
        except FileNotFoundError:
            print(f"Hiba: A fájl nem található: {filepath}")
        except KeyError:
            print(f"Hiba: A CSV fájlnak 'pattern' és 'replacement' oszlopokkal kell rendelkeznie.")
        except Exception as e:
            print(f"Hiba a cserék betöltése során: {e}")

    def insert_spaces_around_symbols(self, text):
        """
        Szóközök beszúrása a '°C' és '%' jelek előtt és mögött.
        Például:
        - "23°C" -> "23 °C"
        - "15%" -> "15 %"
        """
        # Szóköz beszúrása '°C' előtt és után
        text = re.sub(r'(?<!\s)(°C)(?!\s)', r' \1 ', text)
        # Szóköz beszúrása '%' előtt és után
        text = re.sub(r'(?<!\s)(%)(?!\s)', r' \1 ', text)
        # Többszörös szóközök csökkentése egy szóközzé
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def number_to_words(self, number):
        """Egyszerű számok átírása szöveggé magyarul (0-99)."""
        if 0 <= number < 10:
            return self.units[number]
        elif 10 <= number < 20:
            return self.teens[number - 10]
        elif 20 <= number < 100:
            ten, unit = divmod(number, 10)
            if unit == 0:
                return self.tens[ten]
            else:
                return f"{self.tens[ten]}{self.units[unit]}"
        else:
            return str(number)  # Ha túl nagy, visszaadjuk a számot

    def replace_numbers(self, text):
        """Számok átírása szöveggé."""
        def replace_match(match):
            num_str = match.group(0).replace(',', '').replace('.', '')
            try:
                number = int(num_str)
                return self.number_to_words(number)
            except ValueError:
                return num_str  # Ha nem lehet számmá alakítani, visszaadjuk az eredetit

        # Regex a számok felismerésére (egész számok)
        pattern = re.compile(r'\b\d+\b')
        return pattern.sub(replace_match, text)

    def replace_percent(self, text):
        """'%' jel cseréje 'százalék' szóra."""
        pattern = re.compile(r'\b(\d+)\s*%\b')
        return pattern.sub(r'\1 százalék', text)

    def apply_custom_replacements(self, text):
        """Alkalmazza a külső fájlból betöltött szó-pár cseréket."""
        for pattern, replacement in self.custom_replacements.items():
            # Regex minta, amely csak a teljes szavakat cseréli le (case-insensitive)
            regex_pattern = rf'\b{re.escape(pattern)}\b'
            text = re.sub(regex_pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def normalize(self, text):
        """Teljes normalizálási folyamat."""
        # 1. Szóközök beszúrása a '°C' és '%' jelek körül
        text = self.insert_spaces_around_symbols(text)
        # 2. '%' jel cseréje 'százalék' szóra
        text = self.replace_percent(text)
        # 3. Számok cseréje szöveggé
        text = self.replace_numbers(text)
        # 4. Egyéni cserék alkalmazása
        text = self.apply_custom_replacements(text)
        return text

# Példa használat
if __name__ == "__main__":
    # Adjuk meg a külső cseréket tartalmazó fájl elérési útját
    custom_replacements_file = 'custom_replacements.csv'
    normalizer = SimpleHungarianTextNormalizer(custom_replacements_file=custom_replacements_file)
    
    sample_text = """
    **Dátumok és Időpontok:**
    - A találkozó időpontja 2024.01.01
    - Az esemény 9h:30m:45s
    - A futás 2h-15m-30s alatt fejeződött be.
    
    **Pénznemek és Összegek:**
    - A termék ára 299.000 forint.
    - A bevétel 5% volt.
    - A költség 750, és a nyereség 300.
    
    **Számok és Tizedesek:**
    - A hőmérséklet 23°C-ra emelkedett.
    - A pontosság 99.9%.
    
    **Egyéni Cserék:**
    - Az USA gazdasága növekszik.
    - A NASA új küldetést indít.
    - A BBC hírt közöl.
    - Az ABC rövidítés jelentése: American Broadcasting Company.
    - Kyle egy jó barát.
    - Dr. Smith bemutatkozott.
    - Mr. Johnson új projektet indított.
    - Mrs. Davis születésnapi bulit szervezett.
    - Miss Parker diák.
    - Kenty egy népszerű név.
    - Chartman egy híres karakter.
    - Stan a szomszéd.
    """
    
    normalized_text = normalizer.normalize(sample_text)
    print("**Normalizált Szöveg:**\n")
    print(normalized_text)

