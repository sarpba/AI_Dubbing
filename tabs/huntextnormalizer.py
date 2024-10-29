import re
import csv

class HungarianTextNormalizer:
    def __init__(self, custom_replacements_file=None):
        # Szótárak és minták inicializálása
        self.months = {
            'jan': 'január',
            'feb': 'február',
            'márc': 'március',
            'ápr': 'április',
            'máj': 'május',
            'jún': 'június',
            'júl': 'július',
            'aug': 'augusztus',
            'szept': 'szeptember',
            'okt': 'október',
            'nov': 'november',
            'dec': 'december'
        }
        self.weekdays = {
            'h': 'hétfő',
            'k': 'kedd',
            'sze': 'szerda',
            'cs': 'csütörtök',
            'p': 'péntek',
            'szo': 'szombat',
            'v': 'vasárnap'
        }
        self.currency_symbols = {
            '$': 'dollár',
            '€': 'euró',
            '£': 'font',
            '¥': 'jen',
            '₽': 'rubel',
            'HUF': 'forint',
            'Ft': 'forint'
        }
        self.math_symbols = {
            '+': 'meg',
            '-': 'mínusz',
            '*': 'szorozva',
            '/': 'per',
            '=': 'egyenlő',
            '%': 'százalék'
        }

        self.units = {
            '°C': 'fok Celsius',
            '˚C': 'fok Celsius',  # Hozzáadva
            '°F': 'fok Fahrenheit',
            'K': 'kelvin',
            't': 'tonna',
            'q': 'mázsa',
            'kg': 'kilogramm',
            'dkg': 'dekagramm',
            'g': 'gramm',
            'mg': 'milligramm',
            'μg': 'mikrogramm',
            'l': 'liter',
            'dl': 'deciliter',
            'cl': 'centiliter',
            'ml': 'milliliter',
            'km': 'kilométer',
            'm': 'méter',
            'dm': 'deciméter',
            'cm': 'centiméter',
            'mm': 'milliméter',
            'μm': 'mikrométer',
            'nm': 'nanométer',
            'ha': 'hektár',
            'a': 'ár',
            'bar': 'bar',
            'Pa': 'pascal',
            'hPa': 'hektopascal',
            'kPa': 'kilopascal',
            'MPa': 'megapascal',
            's': 'másodperc',
            'min': 'perc',
            'h': 'óra',
            'Hz': 'hertz',
            'kHz': 'kilohertz',
            'MHz': 'megahertz',
            'GHz': 'gigahertz',
            'A': 'amper',
            'V': 'volt',
            'W': 'watt',
            'kW': 'kilowatt',
            'kWh': 'kilowattóra',
            'm²': 'négyzetméter',
            'm³': 'köbméter',
            'Mbps': 'megabit per szekundum',
            'Gbps': 'gigabit per szekundum',
            'B': 'byte',
            'KB': 'kilobyte',
            'MB': 'megabyte',
            'GB': 'gigabyte',
            'TB': 'terabyte',
            # További mértékegységek szükség szerint
        }

        # Betűszavak és egyéb cserék a külső fájlból
        self.custom_replacements = {}
        if custom_replacements_file:
            self.load_custom_replacements(custom_replacements_file)

    def load_custom_replacements(self, filepath):
        """Betölti a szó-pár cseréket egy külső CSV fájlból."""
        try:
            with open(filepath, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    pattern = row['pattern']
                    replacement = row['replacement']
                    self.custom_replacements[pattern] = replacement
        except FileNotFoundError:
            print(f"Hiba: A fájl nem található: {filepath}")
        except KeyError:
            print(f"Hiba: A CSV fájlnak 'pattern' és 'replacement' oszlopokkal kell rendelkeznie.")
        except Exception as e:
            print(f"Hiba a cserék betöltése során: {e}")

    def accent_peculiarity(self, text):
        """Unicode furcsaságok eltávolítása."""
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            '’': "'",
            '“': '"',
            '”': '"',
            '–': '-',  # en dash
            '—': '-',  # em dash
            '˚': '°'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def acronym_phoneme(self, text):
        """Betűszavak fonémikus átírása."""
        def replace_acronyms(match):
            acronym = match.group(0)
            letters = ' '.join([self.letter_to_phoneme(c) for c in acronym])
            return letters

        return re.sub(r'\b[A-ZÁÉÍÓÖŐÚÜŰ]{2,}\b', replace_acronyms, text)

    def letter_to_phoneme(self, letter):
        """Betűk átírása fonémára."""
        phonemes = {
            'A': 'á', 'B': 'bé', 'C': 'cé', 'D': 'dé', 'E': 'é',
            'F': 'ef', 'G': 'gé', 'H': 'há', 'I': 'í', 'J': 'jé',
            'K': 'ká', 'L': 'el', 'M': 'em', 'N': 'en', 'O': 'ó',
            'P': 'pé', 'Q': 'kú', 'R': 'er', 'S': 'es', 'T': 'té',
            'U': 'ú', 'V': 'vé', 'W': 'dupla vé', 'X': 'iksz', 'Y': 'ipszilon', 'Z': 'zé',
            'Á': 'á', 'É': 'é', 'Í': 'í', 'Ó': 'ó', 'Ö': 'ő',
            'Ő': 'ő', 'Ú': 'ú', 'Ü': 'ű', 'Ű': 'ű'
        }
        return phonemes.get(letter.upper(), letter)

    def remove_hyphens_between_words(self, text):
        """Eltávolítja a '-' jeleket két szó vagy egy szám és egy szó között."""
        # Két szó közötti kötőjel eltávolítása
        text = re.sub(r'(\b\w+\b)-(\b\w+\b)', r'\1 \2', text)
        # Szám és szó közötti kötőjel eltávolítása
        text = re.sub(r'(\d+)-(\b\w+\b)', r'\1 \2', text)
        return text

    def amount_money(self, text):
        """Pénznemek átírása."""
        pattern = re.compile(r'(\d+[\d\s,.]*)\s*([€$£¥₽]|HUF|Ft)')
        
        def replace_currency(match):
            amount = match.group(1).strip().replace(' ', '').replace(',', '.')
            currency_symbol = match.group(2)
            currency = self.currency_symbols.get(currency_symbol, currency_symbol)
            
            # Átirányítás, hogy a pénznem a szám után legyen
            # Kivétel, ha a valuta szimbólum után szerepel szöveg, például 'HUF/USD'
            if '/' in currency:
                return f"{amount} {currency}"
            else:
                if '.' in amount:
                    parts = amount.split('.')
                    if len(parts) == 2 and parts[1].isdigit() and parts[1] != '':
                        integer_part, decimal_part = parts
                        integer_part = int(integer_part)
                        decimal_part = int(decimal_part)
                        return f"{self.number_to_hungarian(integer_part)} egész {self.number_to_hungarian(decimal_part)} ezred {currency}"
                    else:
                        # Ha nem megfelelő formátum, visszaadjuk az eredeti szöveget
                        return match.group(0)
                else:
                    try:
                        integer = int(float(amount))
                        return f"{self.number_to_hungarian(integer)} {currency}"
                    except ValueError:
                        return match.group(0)
        
        return pattern.sub(replace_currency, text)

    def date(self, text):
        """Dátumok átírása."""
        pattern = re.compile(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b')
        def replace_date(match):
            day, month, year = match.groups()
            month_name = self.months.get(month.lstrip('0').lower(), f"{month}. hónap")
            return f"{year}. {month_name} {int(day)}."
        return pattern.sub(replace_date, text)

    def timestamp(self, text):
        """Időbélyegek átírása (h:m:s formátum)."""
        pattern = re.compile(r'(\d{1,2})h:(\d{1,2})m:(\d{1,2})s')
        def replace_timestamp(match):
            hours, minutes, seconds = match.groups()
            return f"{int(hours)} óra {int(minutes)} perc {int(seconds)} másodperc"
        return pattern.sub(replace_timestamp, text)

    def time_of_day(self, text):
        """Napi időpontok átírása (h:m vagy h:m:s formátum)."""
        pattern = re.compile(r'\b(\d{1,2})h:(\d{1,2})m(?::(\d{1,2})s)?\b')
        def replace_time(match):
            hours, minutes, seconds = match.groups()
            result = f"{self.number_to_hungarian(int(hours))} óra {self.number_to_hungarian(int(minutes))} perc"
            if seconds:
                result += f" {self.number_to_hungarian(int(seconds))} másodperc"
            return result
        return pattern.sub(replace_time, text)

    def weekday(self, text):
        """Hét napjainak rövidítéseinek átírása."""
        for abbr, full in self.weekdays.items():
            pattern = re.compile(rf'\b{re.escape(abbr)}\b', flags=re.IGNORECASE)
            text = pattern.sub(full, text)
        return text

    def month(self, text):
        """Hónapok rövidítéseinek átírása."""
        for abbr, full in self.months.items():
            pattern = re.compile(rf'\b{re.escape(abbr)}\b', flags=re.IGNORECASE)
            text = pattern.sub(full, text)
        return text

    def ordinal(self, text):
        """Sorszámok átírása (csak nem-tizedes)."""
        pattern = re.compile(r'\b(\d+)\.(?!\d)')
        def replace_ordinal(match):
            number = int(match.group(1))
            ordinal = self.number_to_ordinal(number)
            return ordinal
        return pattern.sub(replace_ordinal, text)

    def number_to_ordinal(self, number):
        """Sorszám számának átírása szöveggé."""
        ordinals = {
            1: 'első',
            2: 'második',
            3: 'harmadik',
            4: 'negyedik',
            5: 'ötödik',
            6: 'hatodik',
            7: 'hetedik',
            8: 'nyolcadik',
            9: 'kilencedik',
            10: 'tizedik',
            11: 'tizenegyedik',
            12: 'tizenkettedik',
            13: 'tizenharmadik',
            14: 'tizennegyedik',
            15: 'tizenötödik',
            16: 'tizenhatodik',
            17: 'tizenhetedik',
            18: 'tizennyolcadik',
            19: 'tizenkilencedik',
            20: 'huszadik',
            21: 'huszonegyedik',
            22: 'huszonkettedik',
            23: 'huszonharmadik',
            24: 'huszonnegyedik',
            25: 'huszonötödik',
            26: 'huszonhatodik',
            27: 'huszonhetedik',
            28: 'huszonnyolcadik',
            29: 'huszonkilencedik',
            30: 'harmincadik',
            31: 'harmincegyedik',
            32: 'harminckettedik',
            33: 'harmincharmadik',
            34: 'harmincnegyedik',
            35: 'harmincötödik',
            36: 'harminchatodik',
            37: 'harminchetedik',
            38: 'harmincnyolcadik',
            39: 'harminckilencedik',
            40: 'negyvenedik',
            41: 'negyvenegyedik',
            42: 'negyvenkettedik',
            43: 'negyvenharmadik',
            44: 'negyvennegyedik',
            45: 'negyvenötödik',
            46: 'negyvenhatodik',
            47: 'negyvenhetedik',
            48: 'negyvennyolcadik',
            49: 'negyvenkilencedik',
            50: 'ötvenedik',
            51: 'ötvenegyedik',
            52: 'ötvenkettedik',
            53: 'ötvenharmadik',
            54: 'ötvennegyedik',
            55: 'ötvenötödik',
            56: 'ötvenhatodik',
            57: 'ötvenhetedik',
            58: 'ötvennyolcadik',
            59: 'ötvenkilencedik',
            60: 'hatvanadik',
            61: 'hatvanegyedik',
            62: 'hatvankettedik',
            63: 'hatvanharmadik',
            64: 'hatvannegyedik',
            65: 'hatvanötödik',
            66: 'hatvanhatodik',
            67: 'hatvanhetedik',
            68: 'hatvannyolcadik',
            69: 'hatvankilencedik',
            70: 'hetvenedik',
            71: 'hetvenegyedik',
            72: 'hetvenkettedik',
            73: 'hetvenharmadik',
            74: 'hetvennegyedik',
            75: 'hetvenötödik',
            76: 'hetvenhatodik',
            77: 'hetvenhetedik',
            78: 'hetvennyolcadik',
            79: 'hetvenkilencedik',
            80: 'nyolcvanadik',
            81: 'nyolcvanegyedik',
            82: 'nyolcvankettedik',
            83: 'nyolcvanharmadik',
            84: 'nyolcvannegyedik',
            85: 'nyolcvanötödik',
            86: 'nyolcvanhatodik',
            87: 'nyolcvanhetedik',
            88: 'nyolcvannyolcadik',
            89: 'nyolcvankilencedik',
            90: 'kilencvenedik',
            91: 'kilencvenegyedik',
            92: 'kilencvenkettedik',
            93: 'kilencvenharmadik',
            94: 'kilencvennegyedik',
            95: 'kilencvenötödik',
            96: 'kilencvenhatodik',
            97: 'kilencvenhetedik',
            98: 'kilencvennyolcadik',
            99: 'kilencvenkilencedik',
            100: 'századik'
        }
        return ordinals.get(number, f"{number}.")

    def special(self, text):
        """Különleges esetek átírása."""
        pattern = re.compile(r'(\d+)/(\d+)')
        def replace_fraction(match):
            numerator, denominator = match.groups()
            return f"{numerator} a {denominator}-ből"
        return pattern.sub(replace_fraction, text)

    def math_symbol(self, text):
        """Matematikai szimbólumok átírása."""
        # Kezeli a +- és -- eseteket
        text = text.replace('+-', ' plusz mínusz ').replace('--', ' mínusz mínusz ')
        
        # Kezeli a '%' szimbólumot
        text = re.sub(r'(\d+)\s*%', lambda m: f"{self.number_to_hungarian(int(m.group(1)))} százalék", text)
        
        # Majd a többi matematikai szimbólumot
        for symbol, word in self.math_symbols.items():
            if symbol == '%':
                continue  # Már kezelve
            # 1. Szimbólumok, amelyek előzőleg szóköz vagy a sztring eleje állnak, és utána szám következik
            pattern1 = re.compile(rf'(^|\s){re.escape(symbol)}(?=\d)', flags=re.IGNORECASE)
            text = pattern1.sub(rf'\1{word} ', text)
            
            # 2. Szimbólumok, amelyek számot követnek, és utána szóköz vagy a sztring vége áll
            pattern2 = re.compile(rf'(?<=\d){re.escape(symbol)}(?=\s|$)', flags=re.IGNORECASE)
            text = pattern2.sub(f" {word}", text)
            
            # 3. Szimbólumok, amelyek önállóan állnak (szóhatárok között)
            pattern3 = re.compile(rf'\b{re.escape(symbol)}\b', flags=re.IGNORECASE)
            text = pattern3.sub(f" {word} ", text)
        
        return text

    def units_of_measurement(self, text):
        """Mértékegységek átírása."""
        # Kivétel kezelése, hogy a 'm' ne legyen átírva, ha időpont része
        pattern = re.compile(r'(\b\d+[\d\s,\.]*)\s*([°˚]?[A-Za-zμ²³]+)\b')
        
        def replace_units(match):
            amount = match.group(1).strip().replace(' ', '').replace(',', '.')
            unit = match.group(2)
            
            # Speciális kezelés a 'm' és 'min' esetében
            if unit.lower() == 'm' and re.search(r'\b\d+h:\d+m\b', text):
                # Ha időpont része, ne cserélje le
                return match.group(0)
            
            unit_full = self.units.get(unit, self.units.get(unit.lower(), unit))
            
            if '.' not in amount:
                try:
                    integer = int(float(amount))
                    return f"{self.number_to_hungarian(integer)} {unit_full}"
                except ValueError:
                    return match.group(0)  # Visszaadjuk az eredeti szöveget, ha hiba
            else:
                parts = amount.split('.')
                if len(parts) == 2 and parts[1].isdigit() and parts[1] != '':
                    integer_part, decimal_part = parts
                    integer_part = int(integer_part)
                    decimal_part = int(decimal_part)
                    return f"{self.number_to_hungarian(integer_part)} egész {self.number_to_hungarian(decimal_part)} tized {unit_full}"
                else:
                    # Ha nincs decimal_part vagy nem szám, visszaadjuk az eredeti szöveget
                    return match.group(0)
        
        return pattern.sub(replace_units, text)

    def number_to_words(self, text):
        """Számok átírása szöveggé (pl. 13 -> tizenhárom)."""
        # Finomított minta a matematikai kontextus kizárására
        pattern = re.compile(
            r'(?<![\^=\+\-\*/])\b(\d+)(?:\.(\d+))?\b(?![\+\-\*/%])'
        )
        
        def replace_number(match):
            integer = int(match.group(1))
            decimal = match.group(2)
            words = self.number_to_hungarian(integer)
            if decimal:
                decimal_length = len(decimal)
                decimal_number = int(decimal)
                decimal_word = self.number_to_hungarian(decimal_number)
                if decimal_length == 1:
                    decimal_unit = 'tized'
                elif decimal_length == 2:
                    decimal_unit = 'század'
                elif decimal_length == 3:
                    decimal_unit = 'ezred'
                elif decimal_length == 4:
                    decimal_unit = 'százezred'
                else:
                    decimal_unit = f"{self.number_to_hungarian(decimal_number)} tized"
                words += f" egész {decimal_word} {decimal_unit}"
            return words
        
        return pattern.sub(replace_number, text)

    def number_to_hungarian(self, number):
        """Egyszerű számok szöveggé alakítása magyarul."""
        # Korlátozott tartomány, bővíthető
        units = ["nulla", "egy", "kettő", "három", "négy", "öt", "hat", "hét", "nyolc", "kilenc"]
        teens = ["tíz", "tizenegy", "tizenkettő", "tizenhárom", "tizennégy", "tizenöt", "tizenhat", "tizenhét", "tizennyolc", "tizenkilencedik"]
        tens = ["", "", "húsz", "harminc", "negyven", "ötven", "hatvan", "hetven", "nyolcvan", "kilencven"]
        hundreds = ["", "száz", "kettőszáz", "háromszáz", "négyszáz", "ötszáz", "hatszáz", "hétszáz", "nyolcszáz", "kilencszáz"]
        thousands = ["", "ezer", "kettő ezer", "három ezer", "négy ezer", "öt ezer", "hat ezer", "hét ezer", "nyolc ezer", "kilenc ezer"]
        millions = ["", "millió", "kettő millió", "három millió", "négy millió", "öt millió", "hat millió", "hét millió", "nyolc millió", "kilenc millió"]

        def convert_chunk(n):
            if n < 10:
                return units[n]
            elif n < 20:
                return teens[n - 10]
            elif n < 100:
                ten, unit = divmod(n, 10)
                return tens[ten] + (units[unit] if unit != 0 else "")
            elif n < 1000:
                hundred, rest = divmod(n, 100)
                return hundreds[hundred] + (" " + convert_chunk(rest) if rest != 0 else "")
            else:
                return str(n)  # Ha túl nagy, visszaadjuk a számot

        if number < 0:
            return "mínusz " + self.number_to_hungarian(-number)
        if number < 1000:
            return convert_chunk(number)
        elif number < 1000000:
            thousand, rest = divmod(number, 1000)
            return convert_chunk(thousand) + " ezer" + (" " + convert_chunk(rest) if rest != 0 else "")
        elif number < 1000000000:
            million, rest = divmod(number, 1000000)
            return convert_chunk(million) + " millió" + (" " + self.number_to_hungarian(rest) if rest != 0 else "")
        else:
            return str(number)  # Ha túl nagy, visszaadjuk a számot

    def apply_custom_replacements(self, text):
        """Alkalmazza a külső fájlból betöltött szó-pár cseréket csak különálló szavakra."""
        for pattern, replacement in self.custom_replacements.items():
            # Regex minta, amely csak a teljes szavakat cseréli le
            regex_pattern = rf'\b{re.escape(pattern)}\b'
            text = re.sub(regex_pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def remove_extra_spaces(self, text):
        """Felesleges szóközök eltávolítása soronként, sortörések megőrzése mellett."""
        lines = text.splitlines()
        normalized_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
        return '\n'.join(normalized_lines)

    def normalize(self, text):
        """Teljes szöveg normalizálása."""
        text = self.remove_hyphens_between_words(text)  # Kötőjelek eltávolítása
        text = self.accent_peculiarity(text)
        text = self.acronym_phoneme(text)
        text = self.amount_money(text)
        text = self.date(text)
        text = self.timestamp(text)
        text = self.time_of_day(text)
        text = self.weekday(text)
        text = self.month(text)
        text = self.ordinal(text)
        text = self.special(text)
        text = self.math_symbol(text)
        text = self.units_of_measurement(text)
        text = self.number_to_words(text)
        text = self.remove_extra_spaces(text)
        # Alkalmazzuk a külső cseréket a végén
        text = self.apply_custom_replacements(text)
        # További normalizálási lépések szükség szerint
        return text

    # A number_to_ordinal és number_to_hungarian metódusok korábban definiáltak



# Példa használatra - Komplett tesztmintaszöveg
if __name__ == "__main__":
    # Adjuk meg a külső cseréket tartalmazó fájl elérési útját
    custom_replacements_file = 'custom_replacements.csv'
    normalizer = HungarianTextNormalizer(custom_replacements_file=custom_replacements_file)
    sample_text = """
**Dátumok és Időpontok:**
- A találkozó időpontja 2024. 1. hónap 1.
- A következő esemény 2024. 12. hónap 31. lesz.
- Az esemény kezdete 9h:30m, és 17h:45m-ig tart.
- A futás 2h:15m:30s alatt fejeződött be.

**Pénznemek és Összegek:**
- A termék ára 1,299 Ft.
- A bevétel €5,000 volt.
- A költség $750, és a nyereség £300.
- Az árfolyam 250 HUF/USD.
- A befektetés ¥10.0.

**Mértékegységek:**
- A hőmérséklet 23°C-ra emelkedett.
- A távolság 15km volt.
- A kapacitás 3.5l.
- A terület 250ha.
- A tömeg 120kg.
- A sebesség 80km/h.

**Matematikai Szimbólumok:**
- Az egyenlet: E = mc^2.
- A százalékos növekedés 15%.
- A műveletek: 5 + 3 = 8, 10 - 2 = 8
- A következő sorozat: 1, 2, 3, 4, 5*.
- Az összetett művelet: 10+-5 = 5

**Betűszavak és Rövidítések:**
- Az USA (United States of America) gazdasága növekszik.
- A NASA (National Aeronautics and Space Administration) új küldetést indít.
- A BBC (British Broadcasting Corporation) hírt közöl.
- Az ABC rövidítés jelentése: American Broadcasting Company.

**Sorszámok:**
- Ez a 1. próba.
- A 2. helyezett.
- A 3. helyen végeztünk.
- A 4. évad kezdődik.
- Az 5. rendezvény.

**Számok és Tizedesek:**
- A hőmérséklet 36.6°C.
- A tőke 12,345 Ft.
- A mérték 0.75l.
- Az időpont 12.30.
- A pontosság 99.9%.

**Egyéb Szimbólumok és Karakterek:**
- Az ( ) zárójelek használata fontos.
- A "idézőjelek" és 'aposztrófok' helyes használata.
- A különféle kötőjelek: – (en dash) és — (em dash).
- A "ﬁ" és "ﬂ" ligatúrák helyett fi és fl.
- Az @ és # szimbólumok használata a közösségi médiában.
- Az & és * szimbólumok matematikai és szöveges kontextusban.
- kyle, dr, mr, mrs, miss, keny, chartman, stan

    """

    normalized_text = normalizer.normalize(sample_text)
    print("**Normalizált Szöveg:**\n")
    print(normalized_text)



